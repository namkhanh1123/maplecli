"""
MapleCLI Chat Client for API Communication
"""
import requests
import json
import re
from typing import List, Dict, Optional, Tuple

from rich.console import Console

class ChatClient:
    """Handles all communication with the OpenAI-compatible API."""
    def __init__(self, api_base: str, api_key: Optional[str], console: Console):
        self.api_base = api_base
        self.api_key = api_key
        self.console = console
        self.timeout = 30
        self.max_retries = 3
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Remove emojis and all # * special characters from text."""
        # Remove emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002600-\U000026FF"  # Miscellaneous Symbols
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub('', text)
        
        # Remove ALL # and * characters completely
        text = text.replace('#', '')
        text = text.replace('*', '')
        
        return text

    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, any]]:
        """Fetches the list of available models from the API."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with self.console.status("[bold green]Fetching models...[/bold green]"):
                response = requests.get(f"{self.api_base}/models", headers=headers, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and isinstance(data["data"], list):
                    models = [model for model in data["data"] if "id" in model]
                    if model_type:
                        return [m for m in models if model_type in m.get("type", [])]
                    return models
                else:
                    self.console.print("[bold red]Error: Unexpected API response format[/bold red]")
                    return []
            else:
                self.console.print(f"[bold red]Error: Failed to fetch models with status code {response.status_code}[/bold red]")
                return []
        except requests.exceptions.RequestException as e:
            self.console.print(f"[bold red]Error: {e}[/bold red]")
            return []

    def stream_chat(self, history: List[Dict[str, str]], model: str, 
                    temperature: float = 0.7, max_tokens: Optional[int] = None) -> Tuple[List[Dict[str, str]], str]:
        """Sends a prompt to the chat API and streams the response."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {"model": model, "messages": history, "stream": True, "temperature": temperature}
        if max_tokens:
            data["max_tokens"] = max_tokens

        assistant_response = ""
        display_buffer = ""
        in_thinking_block = False
        
        print()
        self.console.print("[bold white]┌─[/bold white] [bold magenta]Assistant[/bold magenta]")
        self.console.print("[bold white]|[/bold white]  [dim italic]thinking...[/dim italic]", end="")
        
        for attempt in range(self.max_retries):
            try:
                with requests.post(f"{self.api_base}/chat/completions", headers=headers, json=data, stream=True, timeout=self.timeout) as response:
                    if response.status_code == 200:
                        print("\r" + " " * 80 + "\r", end="", flush=True)
                        self.console.print("[bold white]└─>[/bold white] ", end="")
                        
                        for chunk in response.iter_lines():
                            if chunk:
                                chunk_str = chunk.decode('utf-8')
                                if chunk_str.startswith("data: "):
                                    chunk_str = chunk_str[6:]
                                if chunk_str == "[DONE]":
                                    break
                                try:
                                    json_chunk = json.loads(chunk_str)
                                    if "choices" in json_chunk and len(json_chunk["choices"]) > 0:
                                        content = json_chunk["choices"][0].get("delta", {}).get("content")
                                        if content:
                                            # Check for thinking tags
                                            if '<think>' in content or '<thinking>' in content:
                                                in_thinking_block = True
                                            if '</think>' in content or '</thinking>' in content:
                                                in_thinking_block = False
                                                continue  # Skip this chunk
                                            
                                            # Skip content inside thinking blocks
                                            if in_thinking_block:
                                                assistant_response += content  # Store but don't display
                                                continue
                                            
                                            # Clean the content before displaying
                                            cleaned_content = self.clean_text(content)
                                            assistant_response += cleaned_content
                                            print(cleaned_content, end="", flush=True)
                                except json.JSONDecodeError:
                                    pass
                        print()
                        break
                    else:
                        error_message = f"API request failed with status code {response.status_code}"
                        self.console.print(f"\n[bold red]Error: {error_message}[/bold red]")
                        if attempt >= self.max_retries - 1:
                            raise Exception(error_message)
            except requests.exceptions.RequestException as e:
                self.console.print(f"\n[bold red]Error: {e}[/bold red]")
                if attempt >= self.max_retries - 1:
                    raise
        
        history.append({"role": "assistant", "content": assistant_response})
        return history, assistant_response

    def generate_image(self, prompt: str, model: str = "dall-e-3", size: str = "1024x1024", quality: str = "standard", n: int = 1) -> List[str]:
        """Generates images from text prompt."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {"model": model, "prompt": prompt, "n": n, "size": size, "quality": quality}
        
        try:
            self.console.print("[bold cyan]Generating image...[/bold cyan]")
            response = requests.post(f"{self.api_base}/images/generations", headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                return [img.get("url") or img.get("b64_json") for img in response.json().get("data", [])]
            else:
                self.console.print(f"[bold red]Image generation failed with status {response.status_code}[/bold red]")
                return []
        except requests.exceptions.RequestException as e:
            self.console.print(f"[bold red]Error: {e}[/bold red]")
            return []

    def generate_video(self, prompt: str, model: str = "sora-2") -> Optional[str]:
        """Generates video from text prompt."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {"model": model, "prompt": prompt}
        
        try:
            self.console.print("[bold cyan]Generating video...[/bold cyan]")
            response = requests.post(f"{self.api_base}/videos", headers=headers, json=data, timeout=300)
            if response.status_code == 200:
                result = response.json()
                return result.get("url") or (result.get("data", [{}])[0].get("url") if result.get("data") else None)
            else:
                self.console.print(f"[bold red]Video generation failed with status {response.status_code}[/bold red]")
                return None
        except requests.exceptions.RequestException as e:
            self.console.print(f"[bold red]Error: {e}[/bold red]")
            return None

    def text_to_speech(self, text: str, model: str = "tts-1", voice: str = "alloy", speed: float = 1.0) -> Optional[bytes]:
        """Converts text to speech."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {"model": model, "input": text, "voice": voice, "speed": speed}
        
        try:
            self.console.print("[bold cyan]Generating speech...[/bold cyan]")
            response = requests.post(f"{self.api_base}/audio/speech", headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                return response.content
            else:
                self.console.print(f"[bold red]TTS failed with status {response.status_code}[/bold red]")
                return None
        except requests.exceptions.RequestException as e:
            self.console.print(f"[bold red]Error: {e}[/bold red]")
            return None