from y_client.llm import AssistantAgent, MultimodalConversableAgent


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
            message=f"""Analyze this image for Reddit/social media sharing. Write a detailed description covering:

1. **VISUAL CONTENT**: What's literally shown (people, animals, objects, text overlays, scene)

2. **MEME DETECTION**:
   - Is this a meme? If so, identify the meme format/template (e.g., "Distracted Boyfriend", "Drake Hotline", "Woman Yelling at Cat", "This is Fine", "Expanding Brain")
   - Any text overlays or captions? Quote them exactly
   - Is it using a recognizable meme structure (top text/bottom text, reaction image, screenshot with caption, etc.)?

3. **HUMOR ANALYSIS**:
   - What makes this funny or shareable? (irony, absurdity, relatability, shock value, wholesome, cringe, unexpected twist)
   - Is there a punchline or comedic timing element?
   - Cultural references or inside jokes?

4. **SUBTLE ELEMENTS**:
   - Background details that add to the humor
   - Facial expressions or body language that sell the joke
   - Juxtaposition or contrast that creates comedy
   - Any "hidden" elements viewers might miss on first look

5. **SHAREABILITY FACTORS**:
   - Why would someone share this on Reddit?
   - What emotion does it evoke? (amusement, nostalgia, outrage, wholesomeness, cringe)
   - What communities/subreddits would appreciate this?

Write in English. Be specific about visual details that make this image work as social media content.
<img {image}>""",
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
