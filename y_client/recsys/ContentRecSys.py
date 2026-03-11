import json
from requests import post


class ContentRecSys(object):
    def __init__(self, n_posts=10, visibility_rounds=36):
        """
        Content recommendation system.

        :param n_posts: the number of posts to recommend
        :param visibility_rounds: the number of visibility rounds
        """
        self.name = "ContentRecSys"
        self.params = {
            "limit": n_posts,
            "mode": "default",
            "visibility_rounds": visibility_rounds,
        }

    def add_user_id(self, uid):
        """
        Add user id to the request.

        :param uid: user id
        """
        self.params["uid"] = uid

    def read(self, base_url, user_id, articles=False):
        """
        Read n_posts from the service.

        :param base_url: the base url of the service
        :param articles: whether to return articles or not
        :return: the response from the service
        """
        api_url = f"{base_url}/read"

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        if articles:
            self.params["articles"] = True

        self.params["uid"] = user_id

        st = json.dumps(self.params)

        response = post(f"{api_url}", headers=headers, data=st)

        return response.__dict__["_content"].decode("utf-8")

    def read_mentions(self, base_url):
        """
        Read n_posts from the service.

        :param base_url: the base url of the service
        :return: the response from the service
        """
        api_url = f"{base_url}/read_mentions"

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        st = json.dumps(self.params)
        response = post(f"{api_url}", headers=headers, data=st)

        return response.__dict__["_content"].decode("utf-8")

    def search(self, base_url):
        """
        Search for a query.

        :param base_url: the base url of the service
        :return: the response from the service
        """
        api_url = f"{base_url}/search"

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        st = json.dumps(self.params)
        response = post(f"{api_url}", headers=headers, data=st)

        return response.__dict__["_content"].decode("utf-8")


class ReverseChrono(ContentRecSys):
    def __init__(self, n_posts=10, visibility_rounds=36):
        """
        Reverse chronological content recommendation system.

        :param n_posts: the number of posts to recommend
        :param visibility_rounds: the number of visibility rounds
        """
        super(ReverseChrono, self).__init__(
            n_posts=n_posts, visibility_rounds=visibility_rounds
        )
        self.name = "ReverseChrono"
        self.params = {
            "limit": 10,
            "mode": "rchrono",
            "visibility_rounds": visibility_rounds,
        }


class ReverseChronoPopularity(ContentRecSys):
    def __init__(self, n_posts=10, visibility_rounds=36):
        """
        Reverse chronological popularity content recommendation system.

        :param n_posts: the number of posts to recommend
        :param visibility_rounds: the number of visibility rounds
        """
        super(ReverseChronoPopularity, self).__init__(
            n_posts=n_posts, visibility_rounds=visibility_rounds
        )
        self.name = "ReverseChronoPopularity"
        self.params = {
            "limit": 10,
            "mode": "rchrono_popularity",
            "visibility_rounds": visibility_rounds,
        }


class ReverseChronoFollowers(ContentRecSys):
    def __init__(self, n_posts=10, followers_ratio=0.6, visibility_rounds=36):
        """
        Reverse chronological followers content recommendation system.

        :param n_posts: the number of posts to recommend
        :param followers_ratio: the ratio posts from followers to recommend
        :param visibility_rounds: the number of visibility rounds
        """
        super(ReverseChronoFollowers, self).__init__(
            n_posts=n_posts, visibility_rounds=visibility_rounds
        )
        self.name = "ReverseChronoFollowers"
        self.params = {
            "limit": 10,
            "followers_ratio": followers_ratio,
            "mode": "rchrono_followers",
            "visibility_rounds": visibility_rounds,
        }


class ReverseChronoFollowersPopularity(ContentRecSys):
    def __init__(self, n_posts=10, followers_ratio=0.6, visibility_rounds=36):
        """
        Reverse chronological followers popularity content recommendation system.

        :param n_posts: the number of posts to recommend
        :param followers_ratio: the ratio posts from followers to recommend
        :param visibility_rounds: the number of visibility rounds
        """
        super(ReverseChronoFollowersPopularity, self).__init__(
            n_posts=n_posts, visibility_rounds=visibility_rounds
        )
        self.name = "ReverseChronoFollowersPopularity"
        self.params = {
            "limit": 10,
            "followers_ratio": followers_ratio,
            "mode": "rchrono_followers_popularity",
            "visibility_rounds": visibility_rounds,
        }


class HotRanking(ContentRecSys):
    def __init__(
        self,
        n_posts=10,
        visibility_rounds=36,
        round_decay=12,
        hot_vote_thresh1=3,
        hot_vote_thresh2=8,
        hot_longtail_j1=0.45,
        hot_longtail_j2=0.20,
    ):
        """
        Reddit-style hot ranking content recommendation system.

        Uses logarithmic scoring: log10(max(|score|, 1)) + sign(score) * round / round_decay
        This gives new posts a jumpstart - every round_decay rounds, a post needs ~10x
        more votes to maintain the same position against newer posts.

        :param n_posts: the number of posts to recommend
        :param visibility_rounds: the number of visibility rounds
        :param round_decay: rounds for 10x vote equivalence (default 12)
        """
        super(HotRanking, self).__init__(
            n_posts=n_posts, visibility_rounds=visibility_rounds
        )
        self.name = "HotRanking"
        self.params = {
            "limit": n_posts,
            "mode": "hot",
            "visibility_rounds": visibility_rounds,
            "round_decay": round_decay,
            "hot_vote_thresh1": hot_vote_thresh1,
            "hot_vote_thresh2": hot_vote_thresh2,
            "hot_longtail_j1": hot_longtail_j1,
            "hot_longtail_j2": hot_longtail_j2,
        }


class TopRanking(ContentRecSys):
    def __init__(self, n_posts=10, visibility_rounds=36):
        """
        Reddit-style top ranking content recommendation system.

        Sorts posts by net score (likes - dislikes) without time decay.

        :param n_posts: the number of posts to recommend
        :param visibility_rounds: the number of visibility rounds
        """
        super(TopRanking, self).__init__(
            n_posts=n_posts, visibility_rounds=visibility_rounds
        )
        self.name = "TopRanking"
        self.params = {
            "limit": n_posts,
            "mode": "top",
            "visibility_rounds": visibility_rounds,
        }


class MostCommented(ContentRecSys):
    def __init__(self, n_posts=10, visibility_rounds=36):
        """
        Reddit-style most commented content recommendation system.

        Sorts posts by comment count.

        :param n_posts: the number of posts to recommend
        :param visibility_rounds: the number of visibility rounds
        """
        super(MostCommented, self).__init__(
            n_posts=n_posts, visibility_rounds=visibility_rounds
        )
        self.name = "MostCommented"
        self.params = {
            "limit": n_posts,
            "mode": "most_commented",
            "visibility_rounds": visibility_rounds,
        }
