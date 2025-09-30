from config import EEGFlowConfig
from data.loader import EntropyDataLoader
from models.ft_transformer import FTTransformerTorch2
from training.trainer import EEGTrainer

def main():
    cfg = EEGFlowConfig()
    loader = EntropyDataLoader(cfg.base_path, cfg.rows_per_subject)

    # Example: load one entropy and subject
    g, r = loader.load_entropy(cfg.entropies[0], cfg.segment)
    subj_block = loader.subject_block(g, 0)

    # Define model and trainer
    model = FTTransformerTorch2(n_epochs=10, verbose=True)
    trainer = EEGTrainer(model)

    # Dummy example: you would replace X_train, y_train, X_test, y_test with real splits
    # acc, f1 = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
