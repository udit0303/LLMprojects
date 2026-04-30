import os
import base64
import mimetypes
from pathlib import Path
from mcp.server.fastmcp import FastMCP
import logging
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

# Initialize the FastMCP server
mcp = FastMCP("VisualAnalysisServer")

# Anthropic client reads ANTHROPIC_API_KEY from the environment
client = Anthropic()


@mcp.tool()
def describe_image(file_path: str) -> str:
    """
    Loads an image from a local file path and returns a detailed,
    descriptive paragraph about its content. If the image is of a known
    landmark, work of art, or location, it will be identified by name.

    Args:
        file_path: The absolute path to the image file on the server.

    Returns:
        A single string containing a detailed description of the image,
        or an error message if loading or analysis fails.
    """
    try:
        image_path = Path(file_path)
        if not image_path.is_file():
            return f"Error: File not found at path: {file_path}"

        with open(image_path, "rb") as f:
            image_data = f.read()
        base64_string = base64.b64encode(image_data).decode("utf-8")

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"

        prompt_text = (
            "Analyze this image in detail. Provide a concise, one-paragraph description. "
            "If it is a famous landmark, work of art, or specific location, identify it by name. "
            "Focus on the most important and defining elements in the image that would be useful for a web search. "
            "For example, instead of 'a building', say 'the Eiffel Tower in Paris'. "
            "Do not add any conversational filler; return only the description."
        )

        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": base64_string,
                        },
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }],
        )

        return next(
            (b.text for b in response.content if b.type == "text"), ""
        ).strip()

    except Exception as e:
        return f"Error analyzing image: {e}"


if __name__ == "__main__":
    logging.getLogger("mcp").setLevel(logging.WARNING)
    mcp.run(transport="stdio")
