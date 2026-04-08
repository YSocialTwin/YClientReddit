import json
import threading
import time as pytime
from requests import get, post

__all__ = ["SimulationSlot"]


class SimulationSlot(object):
    def __init__(self, config, client_id=None, heartbeat_interval=5.0, poll_interval=0.25):
        """
        Initialize the SimulationSlot object.

        :param config: the configuration dictionary
        """
        self.base_url = config["servers"]["api"].rstrip("/") + "/"
        self.client_id = str(client_id or "").strip() or "client"
        self.heartbeat_interval = float(heartbeat_interval)
        self.poll_interval = float(poll_interval)
        self._last_heartbeat = 0.0
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = None

        api_url = f"{self.base_url}current_time"

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = get(f"{api_url}", headers=headers)
        data = json.loads(response.__dict__["_content"].decode("utf-8"))

        self.day = data["day"]
        self.slot = data["round"]
        self.id = data["id"]
        self.register_client()

    def get_current_slot(self):
        """
        Get the current slot.

        :return: the current slot, day and id
        """

        api_url = f"{self.base_url}current_time"

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = get(f"{api_url}", headers=headers)
        data = json.loads(response.__dict__["_content"].decode("utf-8"))

        self.day = data["day"]
        self.slot = data["round"]
        self.id = data["id"]

        return self.id, self.day, self.slot

    def _post_json(self, path, payload):
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        return post(f"{self.base_url}{path.lstrip('/')}", headers=headers, data=json.dumps(payload))

    def register_client(self):
        response = self._post_json("register_client", {"client_id": self.client_id})
        data = json.loads(response.__dict__["_content"].decode("utf-8"))
        if response.status_code >= 400:
            raise RuntimeError(f"client registration failed: {data}")
        self.day = data["day"]
        self.slot = data["round"]
        self.id = data["id"]
        self._last_heartbeat = pytime.time()
        self._ensure_heartbeat_worker()
        return data

    def _ensure_heartbeat_worker(self):
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_worker,
            name=f"SimulationSlotHeartbeat-{self.client_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat_worker(self):
        interval = max(0.1, float(self.heartbeat_interval))
        while not self._heartbeat_stop.wait(interval):
            try:
                self.heartbeat(force=True)
            except Exception:
                if self._heartbeat_stop.is_set():
                    return

    def _stop_heartbeat_worker(self):
        self._heartbeat_stop.set()
        worker = self._heartbeat_thread
        if worker and worker.is_alive():
            worker.join(timeout=max(1.0, self.heartbeat_interval))
        self._heartbeat_thread = None

    def heartbeat(self, force=False):
        now = pytime.time()
        if not force and (now - self._last_heartbeat) < self.heartbeat_interval:
            return None
        response = self._post_json("heartbeat", {"client_id": self.client_id})
        data = json.loads(response.__dict__["_content"].decode("utf-8"))
        if response.status_code >= 400:
            raise RuntimeError(f"heartbeat failed: {data}")
        self._last_heartbeat = now
        return data

    def maybe_heartbeat(self):
        return self.heartbeat(force=False)

    def complete_client(self):
        try:
            response = self._post_json("complete_client", {"client_id": self.client_id})
            data = json.loads(response.__dict__["_content"].decode("utf-8"))
            if response.status_code >= 400:
                raise RuntimeError(f"complete_client failed: {data}")
            self.day = data["day"]
            self.slot = data["round"]
            self.id = data["id"]
            return data
        finally:
            self._stop_heartbeat_worker()

    def increment_slot(self):
        """
        Update the current slot.
        """
        current_round_id = int(self.id)
        response = self._post_json(
            "submit_round",
            {
                "client_id": self.client_id,
                "round_id": current_round_id,
                "day": int(self.day),
                "round": int(self.slot),
            },
        )
        data = json.loads(response.__dict__["_content"].decode("utf-8"))
        if response.status_code >= 400 and data.get("error") != "round_mismatch":
            raise RuntimeError(f"submit_round failed: {data}")

        if data.get("error") == "round_mismatch":
            self.day = int(data["day"])
            self.slot = int(data["round"])
            self.id = int(data["id"])
            return

        if data.get("advanced"):
            self.day = int(data["day"])
            self.slot = int(data["round"])
            self.id = int(data["id"])
            return

        while True:
            pytime.sleep(self.poll_interval)
            self.heartbeat(force=True)
            rid, day, slot = self.get_current_slot()
            if int(rid) != current_round_id:
                self.id = int(rid)
                self.day = int(day)
                self.slot = int(slot)
                return
