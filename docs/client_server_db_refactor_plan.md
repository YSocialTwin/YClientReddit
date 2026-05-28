# Client-Server DB Contract Refactor Plan

## Goal

Restore a clear persistence contract across `YClient` and `YClientReddit`:

- the **server** is the only component allowed to read or write the **experiment/social database**
- the **client** may keep a **client-local operational store** only for data that is explicitly client-owned
- opinion dynamics must use the server API for all persisted state

This document is based on the historical structure in `YClient` `main`, where the client already maintained a separate SQLite database for feed and content helper state.

## Historical baseline from `YClient` `main`

`YClient` `main` did **not** implement a pure stateless client. It already had a client-side database layer under `y_client/news_feeds/client_modals.py`.

That client-local database contained:

- `websites`
- `articles`
- `images`
- `agent_custom_prompt`

It was created from `data_schema/database_clean_client.db` and stored under `experiments/{simulation_name}.db`.

Key historical ownership in `YClient` `main`:

- server-owned social state:
  - users
  - follows
  - posts
  - comments/reactions
  - interests
  - rounds
- client-owned helper state:
  - feed cache
  - article cache
  - image cache
  - custom prompt overrides

This means the correct refactor target is **not** “no client persistence at all”. The correct target is:

- no client access to the **server database**
- explicit handling of whether the legacy client DB remains local or is serverized

## What regressed

The regression was not just the removal of `y_client/opinion_dynamics/*`.

The larger contract drift happened because langchain-era code started mixing:

- valid client-local persistence concerns
- invalid direct access to data that should be server-owned

The clearest invalid case was opinion dynamics:

- opinions are part of the simulation state
- opinion rows belong in the server-managed experiment database
- client-side direct reads and writes to `agent_opinion` broke the contract

That path has now been redirected back toward the server API model.

## Current ownership model to enforce

### Server-owned database

The server must exclusively own:

- users and user metadata
- rounds / simulation time
- interests and post-topic mappings
- posts, comments, reactions, follows
- opinion trajectories
- any state that affects simulation truth or cross-client consistency

### Client-local database

A client-local store is still acceptable if it is intentionally scoped to:

- RSS/feed ingestion cache
- article metadata cache
- downloaded image metadata
- temporary enrichment artifacts
- local prompt customization, if that remains a client concern

This store must be treated as an implementation detail of the client, not as simulation truth.

## Inventory from `YClient` `main`

### Clearly legacy client-local DB usage

These files in `YClient` `main` already depended on the client-local DB:

- `y_client/news_feeds/client_modals.py`
- `y_client/news_feeds/feed_reader.py`
- `y_client/clients/client_base.py`
- `y_client/clients/client_web.py`
- `y_client/classes/page_agent.py`
- parts of `y_client/classes/base_agent.py`

Responsibilities they handled:

- provisioning the client DB
- caching websites/articles/images
- selecting local articles or images for page/news posting
- clearing client cache tables during reset
- storing agent custom prompts

### Server-owned responsibilities that must not be client-local

These must remain behind the server contract:

- `agent_opinion`
- `interests`
- `rounds`
- any social graph mutation
- any post/reaction/comment/follow mutation
- any per-user simulation truth

## Refactor target architecture

The clean target is a two-store model with explicit boundaries:

### Store A: Server experiment DB

Accessed only through HTTP API.

Examples:

- `/register`
- `/follow`
- `/post`
- `/comment`
- `/reaction`
- `/set_user_interests`
- `/get_post_topics`
- `/get_post_topics_name`
- `/get_user_opinions`
- `/get_users_opinions`
- `/set_user_opinions`

### Store B: Client operational DB

Accessed only through a narrow local repository layer inside the client.

Examples:

- feed source registration
- article cache lookup
- image cache lookup
- local prompt customization

The key change is not merely where the bytes live. It is that the code must stop treating both stores as interchangeable.

## Recommended refactor direction

Introduce an explicit persistence split in both clients:

1. `SocialServerGateway`
2. `ClientContentStore`

### `SocialServerGateway`

A thin adapter over the existing server API.

It should own:

- user lookup
- interest lookup/update
- post/topic lookup
- opinion read/write
- all simulation mutations

### `ClientContentStore`

A thin adapter over the existing local SQLAlchemy models, if the team wants to retain the client-local DB.

It should own only:

- websites
- articles
- images
- optional client custom prompts

No code outside this adapter should issue local ORM queries.

## Phased implementation plan

### Phase 0: Freeze the contract

Objective:

- define which tables are server-owned and which are client-owned

Steps:

- document the ownership list in both repositories
- ban direct client access to server-owned tables
- add lint or grep-based checks for known forbidden tables such as `agent_opinion`, `rounds`, `interests`, `post`, `follow`, `reaction`

Success criteria:

- one authoritative ownership matrix exists
- opinion state is formally classified as server-owned

### Phase 1: Finish the opinion-dynamics boundary

Objective:

- ensure all opinion persistence and reads go through the server API

Steps:

- restore `y_client/opinion_dynamics/*`
- keep model dispatch in the package, not inline in agents
- remove client reads/writes of `agent_opinion`
- use `/get_user_opinions`, `/get_users_opinions`, `/set_user_opinions`

Success criteria:

- no `agent_opinion` SQL appears in client code
- no client `sqlite3`/ORM path touches opinion tables
- opinion tests pass against API-mocked flows

### Phase 2: Isolate the legacy client-local DB

Objective:

- keep the historical client DB if needed, but make it explicit and contained

Steps:

- create a `ClientContentStore` abstraction
- move all direct `session.query(...)` feed/article/image access behind that adapter
- stop importing `session` broadly across agents and feed readers
- remove ad hoc DB bootstrapping from `client_web.py` and `client_modals.py`
- centralize client-store initialization in one place

Success criteria:

- only the content-store module touches local SQLAlchemy models
- agents no longer issue raw content-cache queries
- page/news behavior is unchanged in regression tests

### Phase 3: Remove server-DB leakage from mixed client flows

Objective:

- identify client code that still assumes the local DB is the experiment DB

Steps:

- audit every `sqlalchemy`, `session.query`, `text(...)`, raw SQL, and engine creation path
- classify each usage as:
  - client-local and allowed
  - server-owned and forbidden
  - ambiguous and needs redesign
- migrate forbidden accesses to API endpoints

Examples of likely redesign areas:

- selecting images/posts from local mixed tables
- client reset logic that deletes records directly
- prompt persistence if it should be shared rather than local

Success criteria:

- every remaining client DB access is explicitly mapped to a client-owned table
- no ambiguous mixed-ownership codepaths remain

### Phase 4: Decide the future of the client-local DB

Objective:

- choose whether to keep or eliminate the local content store

Options:

- keep it as a cache
- replace it with filesystem/JSON cache
- migrate all content storage to the server

Recommendation:

- do not force this decision inside the opinion-dynamics repair
- first restore the ownership boundary, then make a deliberate product decision

Success criteria:

- architecture decision is explicit
- migration plan exists if the local DB is to be retired

## Required API evaluation

Before removing more client DB access, review whether the server already exposes equivalent operations.

For opinion dynamics, the required endpoints already exist.

For feed/content cache functionality, evaluate whether the server should provide:

- website registration/listing
- article lookup by website/date/link
- image lookup by article/url
- random article/image selection
- prompt override CRUD, if shared behavior is desired

If these are not provided, keeping a client-local content store is the lower-risk option.

## Risks

### Risk 1: Overcorrecting into a stateless client

This would erase a historical design choice that was already present in `YClient` `main`.

Impact:

- large unnecessary migration
- degraded offline/local ingestion behavior
- more server API surface than needed

Mitigation:

- separate “remove server DB leakage” from “remove client-local storage”

### Risk 2: Mixed semantics during migration

If some codepaths use the server while others still use local tables for the same concept, behavior will drift.

Mitigation:

- migrate by ownership domain, not by file

### Risk 3: Hidden agent dependencies on local content tables

Page/news agents currently expect direct access to websites/articles/images.

Mitigation:

- introduce repository interfaces first
- change implementations second

## Testing roadmap

### Contract tests

- verify clients never access server-owned tables directly
- grep/lint guard for forbidden table names and direct SQL against them

### Opinion tests

- seed opinions through server API mocks
- update opinions through server API mocks
- verify model dispatch through `y_client.opinion_dynamics`

### Content-store regression tests

- feed ingestion still stores and retrieves articles/images
- page/news selection still works from the client content store
- reset only clears client-owned cache tables

### Integration tests

- run one simulation with server-backed social state and local content cache
- verify posting, commenting, interests, opinions, and page/news flows all still work

## Success criteria

The refactor is successful when all of the following are true:

- both `YClient` and `YClientReddit` have `y_client/opinion_dynamics/*` restored and used for model dispatch
- no client code directly accesses server-owned persistence such as `agent_opinion`, `interests`, `rounds`, posts, follows, or reactions
- any remaining client DB access is limited to explicitly client-owned cache tables
- local DB initialization and access are centralized behind a content-store abstraction
- opinion regression tests pass
- page/news/feed flows retain behavior
- documentation states the ownership boundary unambiguously

## Recommended immediate next step

Do not continue with ad hoc file-by-file cleanup.

The next practical step is:

1. create an ownership matrix for every table currently touched by the clients
2. wrap the remaining legitimate local DB access behind `ClientContentStore`
3. add a forbidden-access check for server-owned tables in both client repositories

That sequence preserves the historical `YClient` design while restoring the client-server contract cleanly.
