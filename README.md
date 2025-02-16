# SwiftSentiment

This tool uses the [Twitter RoBERTa base sentiment model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
to analyze the sentiment of a CSV file containing lines of text.

It is intended to be used for fast sentiment analysis of transcripts of recordings.

The quantized model is included in the repository and the generated apps,
so you don't need to download it separately. The tool does not require an
internet connection and does not send any data outside of your computer.

## Reference papers

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., NAACL 2019)](https://aclanthology.org/N19-1423/)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., arXiv preprint arXiv:1907.11692)](https://arxiv.org/abs/1907.11692)
- [TimeLMs: Diachronic Language Models from Twitter (Loureiro et al., arXiv:2202.03829)](https://arxiv.org/abs/2202.03829)

## Installation

### macOS

At the moment, SwiftSentiment hasn't been [code signed](https://developer.apple.com/documentation/security/code-signing-services) for macOS, which means that when you try to open the `.dmg` file,
and again the first time you try to open the app itself, macOS will show you some slightly alarming messages:

| Message  | Action   |
|----------|----------|
| ![](https://bin.idiot.sh/Ey7AsGFZQgHJXfaOPiqINZtIKjmz7hPVU7qX8GnsAfRh8kW96n.png) | Click 'Done'.
| ![](https://bin.idiot.sh/iZk6GXLOrUpki6mMQI8bZa06KHvPFPguAXYo05RADK8lX3Xc7Y.png) | Open System Settings > Privacy & Security, scroll down, and click 'Open Anyway'. |
| ![](https://bin.idiot.sh/XUibDv991eHEooruKoQyaNxxvRpE5OLZefs8sDDk7lyZR8jgL1.png) | If the app doesn't reopen itself, double click on it again. In this new dialog box, type your password or use TouchID. |

There is no malware in the file and all the code is open source - Apple are just very passionate about security.
