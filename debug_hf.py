from src.sentiment import SentimentModule

# enable debug=True to print raw HF results
s = SentimentModule(
    debug=True,
    use_hf=True,
    hf_model="cardiffnlp/twitter-roberta-base-sentiment",
    device=-1
)

tests = [
    "I'm really disappointed with how things are going.",
    "It's not as bad as yesterday though.",
    "Actually, I think things might work out.",
    "Thanks for helping me figure this out!"
]

for t in tests:
    out = s.score_text(t)
    print("\nTEXT:", t)
    print("RAW SCORES:", out)

