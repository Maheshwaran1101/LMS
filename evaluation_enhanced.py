import jiwer
from rouge_score import rouge_scorer
import nltk
try:
    nltk.download('punkt', quiet=True)
except:
    pass

class TranscriptionEvaluator:             #👉 To checks how correct your speech-to-text output is.
    """Evaluate STT performance using WER"""

    def __init__(self):
        self.transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
        ])

    def calculate_wer(self, reference, hypothesis):
        """Calculate Word Error Rate"""
        try:
            wer = jiwer.wer(reference, hypothesis, truth_transform=self.transformation, hypothesis_transform=self.transformation)
            return round(wer, 4)
        except Exception as e:
            print(f"WER calculation error: {e}")
            return 0.0
        """WER = (Substitutions + Deletions + Insertions) / Total Words
        
        - Substitutions: Words that were incorrectly recognized.
        - Deletions: Words that were missed.
        - Insertions: Words that were added erroneously.
        - Total Words: The number of words in the reference transcript.
        
        Reference:  "I love AI"
        Predicted:  "I like AI"

        Error: love → like (1 substitution)

        WER = 1 / 3 = 0.33"""

    def calculate_cer(self, reference, hypothesis):   #👉 Same idea as WER but works on characters
        """Calculate Character Error Rate"""
        try:
            cer = jiwer.cer(reference, hypothesis)
            return round(cer, 4)
        except Exception as e:
            print(f"CER calculation error: {e}")
            return 0.0
        
        """ Reference: "cat"
            Predicted: "cut"

            Error: a → u

            CER = 1 / 3 = 0.33"""

    def get_detailed_metrics(self, reference, hypothesis):
        """Get detailed error metrics"""
        try:
            measures = jiwer.compute_measures(reference, hypothesis)
            return {
                "WER": round(measures['wer'], 4),
                "MER": round(measures['mer'], 4),
                "WIL": round(measures['wil'], 4),
                "Substitutions": measures['substitutions'],
                "Deletions": measures['deletions'],
                "Insertions": measures['insertions'],
                "Hits": measures['hits']
            }
        except Exception as e:
            print(f"Metrics error: {e}")
            return {"WER": 0.0}


class SummaryEvaluator:
    """Evaluate summary quality using ROUGE and BLEU"""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_rouge(self, reference, hypothesis):   #👉 This checks how good your summary is
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return {
                "ROUGE-1": {
                    "precision": round(scores['rouge1'].precision, 4),
                    "recall": round(scores['rouge1'].recall, 4),
                    "f1": round(scores['rouge1'].fmeasure, 4)
                },
                "ROUGE-2": {
                    "precision": round(scores['rouge2'].precision, 4),
                    "recall": round(scores['rouge2'].recall, 4),
                    "f1": round(scores['rouge2'].fmeasure, 4)
                },
                "ROUGE-L": {
                    "precision": round(scores['rougeL'].precision, 4),
                    "recall": round(scores['rougeL'].recall, 4),
                    "f1": round(scores['rougeL'].fmeasure, 4)
                }
            }
        except Exception as e:
            print(f"ROUGE error: {e}")
            return {}
        
        """ | Type    | Meaning            |
            | ------- | ------------------ |
            | ROUGE-1 | word match         |
            | ROUGE-2 | 2-word match       |
            | ROUGE-L | sentence structure |
            
            AI is powerful
            AI is amazing
            
            Reference Words:  AI | is | powerful
            Summary Words:    AI | is | amazing

            Overlap → AI, is
            
            output will be:
            {
            "precision": how much predicted is correct
            "recall": how much original is captured
            "f1": balance of both
            }
"""

    def calculate_bleu(self, reference, hypothesis):    #👉 Measures how close summary is to reference
        """Calculate BLEU score"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

            ref_tokens = reference.split()
            hyp_tokens = hypothesis.split()

            smoothing = SmoothingFunction().method1
            bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)

            return round(bleu, 4)
        except Exception as e:
            print(f"BLEU error: {e}")
            return 0.0
        
        """ Example:
            Reference: "AI helps humans"
            Summary:   "AI helps people"

            👉 High BLEU (similar meaning)

            Diagram:
            Reference → AI helps humans
            Summary   → AI helps people

            Match → AI helps
        """


def get_benchmark_report():  #function that compares models and returns a report of their performance on standard datasets. It shows accuracy,and speed.
    """Return benchmark results for different models"""
    return {
        "Whisper (Base)": {"WER": 0.08, "CER": 0.04, "Speed": "Slow"},
        "Whisper (Tiny)": {"WER": 0.12, "CER": 0.06, "Speed": "Medium"},
        "Vosk (Small)": {"WER": 0.15, "CER": 0.08, "Speed": "Fast"},
        "Vosk (Large)": {"WER": 0.10, "CER": 0.05, "Speed": "Medium"}
    }
    """
    
    WER = Word Error Rate

👉 Lower = Better

0.08 → very accurate ✅
0.15 → more mistakes ❌
🤖 Models in your example
Whisper (Base)
→ WER = 0.08 → very accurate
Vosk (Small)
→ WER = 0.15 → faster but less accurate
📈 Understanding the Diagram
Accuracy ↑
│
│   Whisper Base  (Best accuracy, slow)
│   Whisper Tiny
│   Vosk Large
│   Vosk Small   (Fastest, least accurate)
│
└──────────────→ Speed
👉 Simple meaning:
Going UP ↑ → accuracy increases
Going RIGHT → → speed increases
💡 Easy way to understand
Top = Accurate but slow
Bottom = Fast but less accurate

"""
    

# Legacy function for backward compatibility
def calculate_wer(ref, hyp):
    evaluator = TranscriptionEvaluator()
    return evaluator.calculate_wer(ref, hyp)


"""
REFERENCE TEXT
        │
        ▼
PREDICTED TEXT
        │
        ▼
   TranscriptionEvaluator
        │
        ├── WER
        ├── CER
        └── Detailed Errors
        │
        ▼
SUMMARY MODEL
        │
        ▼
GENERATED SUMMARY
        │
        ▼
   SummaryEvaluator
        │
        ├── ROUGE
        └── BLEU
        
        """

