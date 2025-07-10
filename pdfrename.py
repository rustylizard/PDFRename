#!/usr/bin/env python3
import os
from datetime import datetime
from PyPDF2 import PdfReader
import asyncio
from openai import AsyncOpenAI, RateLimitError
from pdf2image import convert_from_path
import pytesseract
import warnings
import shutil
import subprocess
import json
import traceback
import sys
import tkinter as tk
from tkinter import ttk
import threading


# Suppress decompression bomb warnings from PIL
warnings.filterwarnings("ignore", category=UserWarning)
possible_tags = ["shopinvoice", "lawyer", "school", "contract", "travel", "health", "insurancework", "lufthansa", "divorce", "tax", "salary","bank","warranty","receipt", "hotel", "germanwings", "car","shipping label","other"]
tag_rules = ["if none of the rules apply, use best choice from my possible tags", "if you find Maryam Mohsin in the text, the filename must start with Maryam_Mohsin","if document contains Gothaer, tag must be insurancework", "if document contains ATPL, tag must be lufthansa", "if it contains Maywald or Familiengericht, tag must be divorce", "if it contains platanus, tag it school", "finanzamt if related to taxes", "hotel invoices should be tagged hotel", "anything containing germanwings should be tagged germanwings","if it contains Finnair, it should be tagged lufthansa","if you cant find anything that applies, apply tag called other", "if its a pilot license, also called ATPL, in the file name put the expiry of the license with day month and year", "shipping label should be shipping label", "bmw should be car"]

def setup_folders(base_path):
    """Create Done and Error folders if they don't exist"""
    done_path = os.path.join(base_path, "../Done")
    error_path = os.path.join(base_path, "../Error")
    
    for path in [done_path, error_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    return done_path, error_path

def init_log_file(log_file):
    """Initialize log file with header if it doesn't exist"""
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("PDF Processing Log\n")
            f.write("="*50 + "\n\n")

def log_operation(log_file, filename, status, details="", cost=0):
    """Log operations to a text file with newest entries on top"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"[{timestamp}] {filename}\n"
        f"Status: {status}\n"
        f"Details: {details}\n"
        f"Cost: ${cost:.6f}\n"
        f"{'-'*40}\n\n"
    )
    
    # Read existing content if file exists
    existing_content = ""
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            # Skip the header (first 3 lines)
            lines = f.readlines()
            if len(lines) > 3:
                existing_content = "".join(lines[3:])
    
    # Write new content followed by existing content
    with open(log_file, "w") as f:
        f.write("PDF Processing Log - Newest First\n")
        f.write("="*40 + "\n\n")
        f.write(log_entry)
        f.write(existing_content)


class ProgressUI:
    """Simple Tkinter progress bar"""

    def __init__(self, total_files: int):
        self.total_files = total_files
        self.processed = 0
        self.root = tk.Tk()
        self.root.title("PDF Rename Progress")
        self.progress = ttk.Progressbar(self.root, length=300, maximum=total_files)
        self.progress.pack(padx=10, pady=10)
        self.label = tk.Label(self.root, text=f"0 / {total_files}")
        self.label.pack(padx=10, pady=10)

    def increment(self):
        """Increment progress in a thread-safe way"""
        self.root.after(0, self._increment)

    def _increment(self):
        self.processed += 1
        self.progress["value"] = self.processed
        self.label.config(text=f"{self.processed} / {self.total_files}")

    def run(self):
        self.root.mainloop()

    def close(self):
        self.root.quit()
        self.root.destroy()

def extract_text_from_pdf(pdf_path):
    """Extract text using PyPDF2 (for text-based PDFs)"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            if reader.is_encrypted:
                print("  - PDF is encrypted, trying to decrypt...")
                reader.decrypt('')  # Try empty password
                
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text.strip()
    except Exception as e:
        print(f"  - Standard extraction failed: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        return ""

def extract_text_with_ocr(pdf_path):
    """Fallback OCR for image-based PDFs"""
    print("  - Attempting OCR extraction...")
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img)
            if page_text.strip():
                text += f"\n--- PAGE {i+1} ---\n{page_text}\n"
        return text.strip()
    except Exception as e:
        print(f"  - OCR failed: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        return ""


async def generate_better_filename(content, current_name, client, max_retries=5):
    """Generate filename using OpenAI API with simple retry on rate limits."""

    tag_list_str = ", ".join(possible_tags)
    tag_rule_str = " , ".join(tag_rules)
    delay = 1

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You're a document assistant. Your job is to:\n"
                            "1. Suggest a concise filename (3-7 words) from this document. Use spaces between words, no special chars except hyphen, no extension. It should not have too many hyphens. It should be nicely capitalized. Include company name if you find.\n"
                            f"2. Select the best matching tag from this list: [{tag_list_str}] Do not give a tag that is not in my list provided.\n\n"
                            f"3. To help you select tags, you must follow the following rules: [{tag_rule_str}].\n"
                            "Only put the other tag after none of the other tags from the list, fits at all"
                            "Respond in this JSON format:\n"
                            "{ \"filename\": \"...\", \"tag\": \"...\" }"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Current filename: {current_name}\nDocument content:\n{content[:10000]}"
                    }
                ],
                temperature=0.2,
                max_tokens=30
            )
            cost = (
                response.usage.prompt_tokens * 0.15 / 1_000_000
            ) + (
                response.usage.completion_tokens * 0.6 / 1_000_000
            )

            print(f"  - Estimated cost: ${cost:.6f}")

            raw_output = response.choices[0].message.content.strip()
            print("  - GPT raw output:")
            print(raw_output)

            if raw_output.startswith("```"):
                raw_output = raw_output.strip("` \n")
                if raw_output.lower().startswith("json"):
                    raw_output = raw_output[4:].strip()

            result = json.loads(raw_output)
            raw_name = result.get("filename", "").strip()
            tag = result.get("tag", "").strip()

            print(raw_name)
            print(tag)
            clean_name = raw_name.replace("\n", " ")
            return f"{clean_name}.pdf", tag, cost

        except RateLimitError as e:
            if attempt < max_retries - 1:
                print(f"  - Rate limit hit, retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay *= 2
                continue
            else:
                print(f"  - Rate limit error: {str(e)}")
        except Exception as e:
            print(f"  - ChatGPT error: {str(e)}")
            traceback.print_exc(file=sys.stdout)
            break

    return None, 0  # Return None and 0 cost on error



def apply_finder_tag(file_path, tag):
    """Apply a Finder tag using the `tag` CLI tool (must be installed via brew)."""
    result = subprocess.run(["/opt/homebrew/bin/tag", "--add", tag, file_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  - Failed to apply tag '{tag}': {result.stderr.strip()}")
    else:
        print(f"  - Tag '{tag}' applied to: {file_path}")


async def process_single_pdf(semaphore, folder_path, filename, client, done_path, error_path, log_file, progress_callback):
    async with semaphore:
        full_path = os.path.join(folder_path, filename)
        print(f"\nProcessing: {filename}")

        original_mtime = os.path.getmtime(full_path)
        success = False
        error_msg = ""
        new_name = filename  # Default to original name
        action = "unchanged"  # Track what happened
        cost = 0  # Initialize cost

        try:
            # Text extraction
            content = await asyncio.to_thread(extract_text_from_pdf, full_path)
            if not content:
                print("  - No text found, attempting OCR...")
                content = await asyncio.to_thread(extract_text_with_ocr, full_path)

            if not content:
                error_msg = "Could not extract text"
                raise Exception(error_msg)

            # Generate new filename
            result = await generate_better_filename(content, filename, client)
            if result is None or len(result) != 3:
                raise Exception("Filename/tag generation failed or malformed response.")
            new_name, selected_tag, cost = result

            if not new_name:
                error_msg = "Name generation failed"
                raise Exception(error_msg)

            # Handle renaming
            if new_name != filename:
                new_path = os.path.join(folder_path, new_name)
                os.rename(full_path, new_path)
                os.utime(new_path, (os.path.getatime(new_path), original_mtime))
                full_path = new_path  # Update reference after rename
                print(f"  - Renamed to: {new_name}")
                action = "renamed"
            else:
                print("  - Name unchanged")
                action = "unchanged"

            success = True

        except Exception as e:
            error_msg = str(e)
            print(f"  - ERROR: {error_msg}")
            traceback.print_exc(file=sys.stdout)

        # Move file and log results
        try:
            if success:
                dest_path = os.path.join(done_path, new_name)
                src_path = os.path.join(folder_path, new_name)
                await asyncio.to_thread(apply_finder_tag, src_path, selected_tag)
                shutil.move(src_path, dest_path)
                log_operation(log_file, filename, "SUCCESS",
                              f"Content extracted, name {action}, moved to Done/{new_name}",
                              cost)
            else:
                dest_path = os.path.join(error_path, filename)
                shutil.move(full_path, dest_path)
                log_operation(log_file, filename, "ERROR",
                              f"Moved to Error/{filename} - {error_msg}",
                              cost)

        except Exception as e:
            error_msg = f"File move failed: {str(e)}"
            print(f"  - {error_msg}")
            traceback.print_exc(file=sys.stdout)
            log_operation(log_file, filename, "ERROR", error_msg, cost)

        progress_callback()


async def process_pdf_folder(folder_path, api_key, progress_callback, max_concurrency=3):
    """Main processing function"""
    client = AsyncOpenAI(api_key=api_key)
    done_path, error_path = setup_folders(folder_path)
    log_file = os.path.join(folder_path, "processing_log.log")
    log_file = "./processing_log.log"

    # Initialize log file
    init_log_file(log_file)

    # Check directory permissions
    try:
        test_file = os.path.join(folder_path, "test_write.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"Error: Cannot write to directory {folder_path}: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        exit(1)

    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            tasks.append(asyncio.create_task(
                process_single_pdf(semaphore, folder_path, filename, client,
                                   done_path, error_path, log_file,
                                   progress_callback)))

    await asyncio.gather(*tasks)

def run_with_progress(folder_path: str, api_key: str):
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    try:
        ui = ProgressUI(len(pdf_files))
    except tk.TclError:
        print("No display available for Tkinter progress bar.")
        asyncio.run(process_pdf_folder(folder_path, api_key, lambda: None))
        return

    def runner():
        asyncio.run(process_pdf_folder(folder_path, api_key, ui.increment))
        ui.root.after(0, ui.close)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    ui.run()
    thread.join()


if __name__ == "__main__":
    # Get inputs
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        exit(1)

    folder_path = "./Inbox"

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        exit(1)

    run_with_progress(folder_path, api_key)
