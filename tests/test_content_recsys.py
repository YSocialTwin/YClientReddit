"""Content recommender classes: request parameters sent to the server ``/read`` endpoint."""

import copy
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import y_client.recsys as recsys  # noqa: E402

base_module = sys.modules["y_client.recsys.ContentRecSys"]

EXPECTED_MODES = {
    "ContentRecSys": "default",
    "ReverseChrono": "rchrono",
    "ReverseChronoPopularity": "rchrono_popularity",
    "ReverseChronoFollowers": "rchrono_followers",
    "ReverseChronoFollowersPopularity": "rchrono_followers_popularity",
    "ReverseChronoComments": "rchrono_comments",
    "CommonInterests": "common_interests",
    "CommonUserInterests": "common_user_interests",
    "SimilarUsersReactions": "similar_users_react",
    "SimilarUsersPosts": "similar_users_posts",
    "HotRanking": "hot",
    "TopRanking": "top",
    "MostCommented": "most_commented",
}


def test_registry_lists_every_class():
    assert set(recsys.CONTENT_RECSYS) == set(EXPECTED_MODES)


@pytest.mark.parametrize("name,mode", sorted(EXPECTED_MODES.items()))
def test_class_exists_and_sends_mode(name, mode):
    inst = getattr(recsys, name)()
    assert inst.name == name
    assert inst.params["mode"] == mode


@pytest.mark.parametrize("name", sorted(EXPECTED_MODES))
def test_n_posts_is_honoured(name):
    assert getattr(recsys, name)(n_posts=25).params["limit"] == 25


def test_articles_flag_does_not_stick():
    inst = recsys.ReverseChrono()
    with mock.patch.object(base_module, "post") as post:
        post.return_value.__dict__["_content"] = b"[]"
        inst.read("http://x", 1, articles=True)
        assert json.loads(post.call_args.kwargs["data"])["articles"] is True
        inst.read("http://x", 1)
        assert "articles" not in json.loads(post.call_args.kwargs["data"])


def test_agents_get_independent_copies():
    shared = recsys.ReverseChrono()
    a, b = copy.deepcopy(shared), copy.deepcopy(shared)
    a.add_user_id(1)
    b.add_user_id(2)
    assert a.params["uid"] == 1 and b.params["uid"] == 2
