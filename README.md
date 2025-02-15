# SwiftSentiment

This tool uses the [Twitter RoBERTa base sentiment model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
to analyze the sentiment of a CSV file containing lines of text.

It is intended to be used for fast sentiment analysis of transcripts of recordings.

The quantized model is included in the repository and the generated apps,
so you don't need to download it separately. The tool does not require an
internet connection and does not send any data outside of your computer.