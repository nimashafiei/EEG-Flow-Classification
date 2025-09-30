from pathlib import Path
import pandas as pd

class EntropyDataLoader:
    def __init__(self, base_path: str, rows_per_subject: int):
        self.base = Path(base_path)
        self.rows_per_subject = rows_per_subject

    @staticmethod
    def _subj_block(df: pd.DataFrame, s: int, rows: int) -> pd.DataFrame:
        return df.iloc[s*rows:(s+1)*rows, :].copy()

    def load_entropy(self, ent: str, segment: str):
        g = pd.read_csv(self.base / f"{ent}_insight_{segment}_game.csv", header=None).add_prefix(f"{ent}_")
        r = pd.read_csv(self.base / f"{ent}_insight_rest.csv", header=None).add_prefix(f"{ent}_")
        return g, r

    def subject_block(self, df: pd.DataFrame, subj_idx: int) -> pd.DataFrame:
        return self._subj_block(df, subj_idx, self.rows_per_subject)

    def count_subjects(self, entropies, segment: str) -> int:
        counts = []
        for ent in entropies:
            g, r = self.load_entropy(ent, segment)
            counts.append(min(len(g)//self.rows_per_subject, len(r)//self.rows_per_subject))
        return min(counts)
