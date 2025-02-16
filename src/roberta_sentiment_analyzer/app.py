import asyncio
import os
from typing import Dict
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification
import csv


class SwiftSentiment(toga.App):
    def startup(self):
        # Load the model on startup
        path_to_model_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "resources/twitter-roberta-base-sentiment-latest-quantized",
            )
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            path_to_model_dir, local_files_only=True
        )
        model = ORTModelForSequenceClassification.from_pretrained(
            path_to_model_dir, local_files_only=True
        )
        self.classifier = pipeline(
            "text-classification", model=model, tokenizer=self.tokenizer
        )

        # Create main box
        main_box = toga.Box(style=Pack(direction=COLUMN, padding=10))

        # Create title label
        title_label = toga.Label(
            "SwiftSentiment",
            style=Pack(padding=(0, 0, 10, 0), font_size=18),
        )
        intro_label = toga.Label(
            "This tool uses the Twitter RoBERTa base sentiment model to analyze the sentiment of a transcript.\nSelect a CSV file with a single line of text in each row to get started.",
            style=Pack(padding=(0, 0, 10, 0)),
        )

        # Create file selection button and label
        file_selection_box = toga.Box(
            style=Pack(direction=ROW, padding=(0, 0, 10, 0), alignment="center")
        )

        file_button = toga.Button(
            "Select CSV file", on_press=self.select_file, style=Pack(padding=5)
        )

        self.file_path_label = toga.Label(
            "No file selected", style=Pack(padding=(0, 5), flex=1)
        )

        # Create button box for analyze and download buttons
        button_box = toga.Box(
            style=Pack(direction=ROW, padding=(0, 0, 10, 0), alignment="center")
        )

        # Create analyze button
        self.analyze_button = toga.Button(
            "Analyze sentiment", on_press=self.analyze_sentiment, style=Pack(padding=5)
        )
        self.analyze_button.enabled = False

        # Create download button
        self.download_button = toga.Button(
            "Save results", on_press=self.save_results, style=Pack(padding=5)
        )
        self.download_button.enabled = False

        # Create table
        self.results_table = toga.Table(
            headings=["Text", "Sentiment", "Confidence"], style=Pack(flex=1)
        )

        # Add widgets to main box
        main_box.add(title_label)
        main_box.add(intro_label)
        main_box.add(file_selection_box)
        file_selection_box.add(file_button)
        file_selection_box.add(self.file_path_label)

        button_box.add(self.analyze_button)
        button_box.add(self.download_button)
        main_box.add(button_box)

        main_box.add(self.results_table)

        # Create main window
        self.main_window = toga.MainWindow(title=self.formal_name, size=(800, 600))
        self.main_window.content = main_box
        self.main_window.show()

    async def select_file(self, widget):
        try:
            fname = await self.main_window.open_file_dialog(
                title="Select CSV file", file_types=["csv"]
            )

            if fname is not None:
                self.current_file = fname
                self.file_path_label.text = str(fname)
                self.analyze_button.enabled = True
                self.download_button.enabled = False  # Disable download button
                self.results_table.data = []  # Clear the table
                self.results = None  # Clear stored results

        except Exception as e:
            self.main_window.error_dialog("File Selection Error", str(e))

    async def analyze_text_in_chunks(
        self, text: str, max_length: int = 480
    ) -> Dict[str, float]:
        """
        Analyze sentiment of a long text by breaking it into chunks and averaging results.
        Using max_length of 480 to leave room for special tokens ([CLS], [SEP], etc.)
        """
        # Get the encoding without truncation first
        encoding = self.tokenizer(text, truncation=False, add_special_tokens=False)
        input_ids = encoding["input_ids"]

        # Split into chunks
        chunks = []
        current_chunk = []
        current_length = 0

        for token_id in input_ids:
            if current_length >= max_length:
                # Convert chunk to text and add to chunks list
                chunk_text = self.tokenizer.decode(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_length = 0

            current_chunk.append(token_id)
            current_length += 1

        # Add the last chunk if it exists
        if current_chunk:
            chunk_text = self.tokenizer.decode(current_chunk)
            chunks.append(chunk_text)

        # Analyze each chunk
        chunk_analyses = []
        for chunk in chunks:
            try:
                result = self.classifier(chunk)[0]
                chunk_analyses.append(result)
            except Exception as e:
                continue

        if not chunk_analyses:
            return {"label": "neutral", "score": 1.0}  # Default if all chunks fail

        # Aggregate results
        sentiment_scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}

        for analysis in chunk_analyses:
            sentiment_scores[analysis["label"]] += analysis["score"]

        # Average the scores
        total_chunks = len(chunk_analyses)
        for label in sentiment_scores:
            sentiment_scores[label] /= total_chunks

        # Determine overall sentiment
        max_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])

        return {"label": max_sentiment[0], "score": max_sentiment[1]}

    async def analyze_sentiment(self, widget):
        try:
            # Update button text and disable during analysis
            original_text = self.analyze_button.text
            self.analyze_button.text = "Analyzing..."
            self.analyze_button.enabled = False
            self.download_button.enabled = False

            # Wait for button text to update
            await asyncio.sleep(0.1)

            # Read input CSV
            with open(self.current_file, "r", newline="") as infile:
                reader = csv.reader(infile)
                rows = list(reader)

            # Process each line and get sentiment
            self.results = []
            for i, row in enumerate(rows):
                if row:  # Skip empty rows
                    text = row[0]  # Get first column
                    sentiment = self.classifier(text)[0]  # Get sentiment

                    # Add to results: [original text, label, score]
                    self.results.append(
                        [
                            text,
                            sentiment["label"],
                            f"{sentiment['score'] * 100:.2f}%",
                        ]
                    )

                    # Update UI every 10 rows
                    if i % 10 == 0:
                        self.results_table.data = self.results
                        await asyncio.sleep(0.1)

            # Analyze the sentiment of the entire transcript and add to results
            text = "\n".join(row[0] for row in rows)
            overall_sentiment = await self.analyze_text_in_chunks(text)
            self.results.append(
                [
                    "Overall Transcript",
                    overall_sentiment["label"],
                    f"{overall_sentiment['score'] * 100:.2f}%",
                ]
            )

            # Update the table with results
            self.results_table.data = self.results

            # Enable download button
            self.download_button.enabled = True

            self.main_window.info_dialog(
                "Success", "Analysis complete! You can now save the results."
            )

        except Exception as e:
            self.main_window.error_dialog("Analysis error", str(e))
        finally:
            # Restore button text and enable
            self.analyze_button.text = original_text
            self.analyze_button.enabled = True

    async def save_results(self, widget):
        try:
            if not self.results:
                return

            # Get save location from user
            save_path = await self.main_window.save_file_dialog(
                title="Save Results",
                suggested_filename=f"{self.current_file.stem}_analyzed.csv",
            )

            if save_path:
                # Write results to CSV
                with open(save_path, "w", newline="") as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(["Text", "Sentiment", "Confidence"])
                    writer.writerows(self.results)

                self.main_window.info_dialog(
                    "Success", f"Results saved to:\n{save_path}"
                )

        except Exception as e:
            self.main_window.error_dialog("Save Error", str(e))


def main():
    return SwiftSentiment()
