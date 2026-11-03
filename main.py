import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else genai.Client()


DEFAULT_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
]

def send_chat_message(user_input: str, previous_interaction_id: str | None = None, model: str = "gemini-3-flash-preview"):
    """Sends a message using the Interactions API, automatically handling server load retries."""
    for attempt_model in [model] + [m for m in DEFAULT_MODELS if m != model]:
        for attempt in range(2):
            try:
                interaction = client.interactions.create(
                    model=attempt_model,
                    input=user_input,
                    previous_interaction_id=previous_interaction_id,
                )
                return interaction, attempt_model
            except ServerError as e:

                time.sleep(1)
            except Exception as e:

                if "503" in str(e):
                    time.sleep(1)
                    continue
                raise e

    raise RuntimeError("All models are currently experiencing high demand. Please try again in a few moments.")

def main():
    print("=== Gemini Chat CLI ===")
    print("Type 'exit' or 'quit' to end the session.\n")

    current_interaction_id = None
    current_model = "gemini-3-flash-preview"

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            interaction, used_model = send_chat_message(
                user_input=user_input,
                previous_interaction_id=current_interaction_id,
                model=current_model
            )

            current_interaction_id = interaction.id
            current_model = used_model

            print(f"\nGemini: {interaction.output_text}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()
