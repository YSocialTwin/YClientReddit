from __future__ import annotations

import argparse
import importlib
import json
import sqlite3
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def query_one(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> dict | None:
    conn.row_factory = sqlite3.Row
    row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


def query_counts(conn: sqlite3.Connection) -> dict:
    tables = [
        "memory_interaction_events",
        "memory_items",
        "memory_social_cards",
        "memory_thread_cards",
        "memory_community_digests",
    ]
    counts = {}
    for table in tables:
        try:
            row = query_one(conn, f"select count(*) as count from {table}")
            counts[table] = int(row["count"]) if row else 0
        except sqlite3.OperationalError:
            counts[table] = 0
    return counts


def latest_comment_event(conn: sqlite3.Connection) -> dict | None:
    return query_one(
        conn,
        """
        select id, run_id, round_id, actor_user_id, target_user_id, thread_root_id, target_post_id
        from memory_interaction_events
        where event_type = 'comment'
          and target_user_id is not null
          and thread_root_id is not null
        order by id desc
        limit 1
        """,
    )


def load_user(conn: sqlite3.Connection, user_id: int) -> dict | None:
    return query_one(
        conn,
        """
        select id, username, email
        from user_mgmt
        where id = ?
        limit 1
        """,
        (int(user_id),),
    )


def import_agent(client_root: Path):
    sys.path.insert(0, str(client_root))
    module = importlib.import_module("y_client.classes.base_agent")
    return module.Agent


def build_agent(agent_cls, config: dict, prompts: dict, agent_record: dict):
    agent = agent_cls(
        name=agent_record["name"],
        email=agent_record["email"],
        load=True,
        config=config,
    )
    agent.set_prompts(prompts)
    return agent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--agents-file", required=True)
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    client_root = Path(args.client_root).resolve()
    config_path = Path(args.config).resolve()
    prompts_path = Path(args.prompts).resolve()
    agents_path = Path(args.agents_file).resolve()
    db_path = Path(args.db).resolve()

    config = load_json(config_path)
    prompts = load_json(prompts_path)
    agents_payload = load_json(agents_path)

    conn = sqlite3.connect(str(db_path))
    counts = query_counts(conn)
    sample = latest_comment_event(conn)
    result = {
      "db": str(db_path),
      "counts": counts,
      "sample_event": sample,
      "reply_context": None,
      "thread_browse_context": None,
      "post_style_context": None,
    }

    if not sample:
        print(json.dumps(result, indent=2))
        return 0

    agent_cls = import_agent(client_root)
    actor_user = load_user(conn, int(sample["actor_user_id"]))
    other_user = load_user(conn, int(sample["target_user_id"]))
    if actor_user is None:
        raise RuntimeError(f"Actor user {sample['actor_user_id']} not found in database")

    actor_record = None
    for record in agents_payload.get("agents", []):
        if (
            str(record.get("name") or "").strip() == str(actor_user.get("username") or "").strip()
            and str(record.get("email") or "").strip() == str(actor_user.get("email") or "").strip()
        ):
            actor_record = record
            break
    if actor_record is None:
        actor_record = {"name": actor_user["username"], "email": actor_user["email"]}

    agent = build_agent(agent_cls, config, prompts, actor_record)
    other_name = other_user.get("username") if isinstance(other_user, dict) else None

    reply_text, reply_meta = agent._memory_build_reply_context(
        query_text="follow up on our discussion",
        other_user_id=int(sample["target_user_id"]),
        thread_root_id=int(sample["thread_root_id"]),
        other_username=other_name,
        round_id=int(sample["round_id"]),
    )
    browse_text, browse_meta = agent._memory_build_thread_browse_context(
        thread_root_id=int(sample["thread_root_id"]),
        tid=int(sample["round_id"]),
    )
    style_text, style_meta = agent._memory_build_post_style_context(tid=int(sample["round_id"]))

    result["reply_context"] = {
        "text": reply_text,
        "meta": reply_meta,
        "length": len(reply_text or ""),
    }
    result["thread_browse_context"] = {
        "text": browse_text,
        "meta": browse_meta,
        "length": len(browse_text or ""),
    }
    result["post_style_context"] = {
        "text": style_text,
        "meta": style_meta,
        "length": len(style_text or ""),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
