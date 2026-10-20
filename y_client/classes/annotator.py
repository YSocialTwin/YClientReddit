from y_client.llm import AssistantAgent, MultimodalConversableAgent
from y_client import prompt_store


class Annotator(object):
    def __init__(self, config):
        self.config_list = [
            {
                "model": config["model"],
                "base_url": config["url"],
                "timeout": 10000,
                "api_type": "open_ai",
                "api_key": config["api_key"],
                "price": [0, 0],
            }
        ]

        self.image_agent = MultimodalConversableAgent(
            name="image-explainer",
            max_consecutive_auto_reply=1,
            llm_config={
                "cache_seed": None,  # Disable AutoGen's caching
                "config_list": self.config_list,
                "temperature": config['temperature'],
                "max_tokens": config['max_tokens'],
            },
            human_input_mode="NEVER",
        )

        self.user_proxy = AssistantAgent(
            name="User_proxy",
            max_consecutive_auto_reply=0,
        )

    def annotate(self, image):

        self.user_proxy.initiate_chat(
            self.image_agent,
            silent=True,
            message=prompt_store.prompt(
                        "annotator__annotate__1",
                        image=image,
                    ),
        )

        payload = self.image_agent.chat_messages[self.user_proxy][-1]["content"]
        if isinstance(payload, str):
            res = payload
        elif isinstance(payload, list):
            text_chunks = []
            for item in payload:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        text_chunks.append(text.strip())
                elif isinstance(item, str) and item.strip():
                    text_chunks.append(item.strip())
            res = "\n".join(text_chunks).strip()
        elif isinstance(payload, dict):
            res = str(payload.get("text") or "").strip()
        else:
            res = str(payload or "").strip()
        return res
