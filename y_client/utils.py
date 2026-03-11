import json
import random
import re

import faker

try:
    from y_client import Agent
except:
    from y_client.classes.base_agent import Agent


_MAX_USERNAME_LEN = 15
_FORUM_ALLOWED_RE = re.compile(r"[^a-z0-9_]+")
_MICROBLOG_ALLOWED_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_REDDIT_ADJECTIVES = (
    "sleepy",
    "cursed",
    "chaotic",
    "spicy",
    "dusty",
    "awkward",
    "unhinged",
)
_REDDIT_NOUNS = (
    "potato",
    "badger",
    "toaster",
    "gremlin",
    "ferret",
    "waffle",
    "goblin",
)
_REDDIT_TOPICS = (
    "linux",
    "ai",
    "gaming",
    "books",
    "coffee",
    "cycling",
    "metal",
    "cooking",
)
_REDDIT_ROLES = (
    "lurker",
    "mod",
    "enjoyer",
    "skeptic",
    "nerd",
    "fan",
)
_REDDIT_VERBS = ("scroll", "meme", "rant", "debate", "doom")
_REDDIT_PREFIXES = ("actual", "real", "justa", "defnot", "probnot")


def _sanitize_forum_username(raw: str) -> str:
    if not raw:
        return ""
    cleaned = raw.strip().lower().replace(" ", "").replace("-", "_")
    cleaned = _FORUM_ALLOWED_RE.sub("", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:_MAX_USERNAME_LEN]


def _sanitize_microblogging_username(raw: str) -> str:
    if not raw:
        return ""
    cleaned = "".join(str(raw).split())
    cleaned = _MICROBLOG_ALLOWED_RE.sub("", cleaned)
    return cleaned[:_MAX_USERNAME_LEN]


def _ensure_unique_name(base: str, used_names, min_len: int = 1) -> str:
    base = (base or "user")[:_MAX_USERNAME_LEN]
    if len(base) < min_len:
        base = "user"
    if base not in used_names and len(base) >= min_len:
        return base

    counter = 1
    while True:
        suffix = str(counter)
        prefix = base[: max(0, _MAX_USERNAME_LEN - len(suffix))]
        candidate = f"{prefix}{suffix}"
        if len(candidate) >= min_len and candidate not in used_names:
            return candidate
        counter += 1


def _generate_reddit_username_candidate() -> str:
    word = random.choice(_REDDIT_NOUNS + _REDDIT_TOPICS)
    role = random.choice(_REDDIT_ROLES)
    templates = (
        f"{random.choice(_REDDIT_ADJECTIVES)}_{random.choice(_REDDIT_NOUNS)}"
        f"{random.randint(10, 99)}",
        f"{random.choice(_REDDIT_NOUNS)}{random.choice(_REDDIT_NOUNS)}"
        f"{random.randint(10, 99)}",
        f"{random.choice(_REDDIT_TOPICS)}_{role}",
        f"{random.choice(_REDDIT_VERBS)}ing_{random.choice(_REDDIT_TOPICS)}",
        f"{random.choice(_REDDIT_PREFIXES)}{word}_{random.randint(9, 26)}",
        f"u_{word}{random.randint(10, 99)}",
        f"{word}_alt{random.randint(1, 999)}",
        f"not_{word}_{role}",
    )
    return random.choice(templates)


def generate_username_by_type(fake, gender, username_type, used_names):
    username_type = (username_type or "microblogging").strip().lower()
    if username_type not in {"forum", "microblogging"}:
        username_type = "microblogging"

    if username_type == "forum":
        for _ in range(200):
            # Prefer explicit Reddit-style patterns over generic faker usernames.
            candidate = _sanitize_forum_username(_generate_reddit_username_candidate())
            if len(candidate) < 3:
                continue
            if candidate not in used_names:
                used_names.add(candidate)
                return candidate
        for _ in range(60):
            candidate = _sanitize_forum_username(fake.user_name())
            if len(candidate) < 3:
                continue
            if candidate not in used_names:
                used_names.add(candidate)
                return candidate
        candidate = _ensure_unique_name("user", used_names, min_len=3)
        used_names.add(candidate)
        return candidate

    for _ in range(120):
        if gender == "male":
            raw_name = fake.name_male()
        else:
            raw_name = fake.name_female()
        candidate = _sanitize_microblogging_username(raw_name)
        if len(candidate) < 3:
            continue
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
    candidate = _ensure_unique_name("User", used_names, min_len=3)
    used_names.add(candidate)
    return candidate


def generate_user(config, owner=None, username_type=None, used_names=None):
    """
    Generate a fake user.
    """

    locales = json.load(open("config_files/nationality_locale.json"))
    try:
        nationality = random.sample(
            config["agents"]["nationalities"],
            min(1, len(config["agents"]["nationalities"])),
        )[0]
    except:
        nationality = "American"

    gender = random.sample(["male", "female"], 1)[0]

    fake = faker.Faker(locales.get(nationality, "en_US"))
    local_used_names = (
        used_names if isinstance(used_names, set) else set(used_names or [])
    )
    # Hardcode to "forum" for Reddit-style usernames
    resolved_type = "forum"
    name = generate_username_by_type(fake, gender, resolved_type, local_used_names)

    email = f"{name}@{fake.free_email_domain()}"
    political_leaning = fake.random_element(
        elements=(config["agents"]["political_leanings"])
    )
    age = fake.random_int(
        min=config["agents"]["age"]["min"], max=config["agents"]["age"]["max"]
    )
    interests = fake.random_elements(
        elements=set(config["agents"]["interests"]),
        length=fake.random_int(
            min=config["agents"]["n_interests"]["min"],
            max=config["agents"]["n_interests"]["max"],
        ),
    )

    toxicity = fake.random_element(elements=(config["agents"]["toxicity_levels"]))

    language = fake.random_element(elements=(config["agents"]["languages"]))

    ag_type = fake.random_element(elements=(config["agents"]["llm_agents"]))
    pwd = fake.password()

    big_five = {
        "oe": fake.random_element(elements=(config["agents"]["big_five"]["oe"])),
        "co": fake.random_element(elements=(config["agents"]["big_five"]["co"])),
        "ex": fake.random_element(elements=(config["agents"]["big_five"]["ex"])),
        "ag": fake.random_element(elements=(config["agents"]["big_five"]["ag"])),
        "ne": fake.random_element(elements=(config["agents"]["big_five"]["ne"])),
    }

    education_level = fake.random_element(
        elements=(config["agents"]["education_levels"])
    )

    try:
        round_actions = fake.random_int(
            min=config["agents"]["round_actions"]["min"],
            max=config["agents"]["round_actions"]["max"],
        )
    except:
        round_actions = 3

    api_key = config["servers"]["llm_api_key"]

    agent = Agent(
        name=name,
        pwd=pwd,
        email=email,
        age=age,
        ag_type=ag_type,
        leaning=political_leaning,
        interests=list(interests),
        config=config,
        big_five=big_five,
        language=language,
        education_level=education_level,
        owner=owner,
        round_actions=round_actions,
        gender=gender,
        nationality=nationality,
        toxicity=toxicity,
        api_key=api_key,
        is_page=0,
    )

    if not hasattr(agent, "user_id"):
        print(f"Agent creation failed: {name} - no user_id assigned")
        return None

    return agent
