import pandas as pd
import numpy as np

draft = pd.read_csv("clean/draft_nba.csv")

# Draft data

table = draft.to_html()

print(table)