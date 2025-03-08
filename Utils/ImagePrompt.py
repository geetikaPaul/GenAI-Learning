import base64

#useful for Pixtral, OpenAI
class ImageLoaderBase64:

    def __init__(self, user_prompt: str, image_file_path: str):
        with open(image_file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        image_extension = image_file_path.split('.')[-1].lower()
        if image_extension not in ['jpeg', 'jpg', 'png']:
            raise ValueError("Unsupported image format. Please use JPEG or PNG.")

        mime_type = "image/jpeg" if image_extension in ['jpeg', 'jpg'] else "image/png"

        self.encoded_message_with_image = [
            {"type": "text", "text": f"{user_prompt}"},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}",
                "detail": "high",
            },
            },
        ]