from config.setup_config import Config
from manager import Manager

if __name__ == "__main__":
    cfg = Config().cfg
    manager = Manager(cfg)
    df = manager.get_data(chunksize=cfg.data.chunksize)
    df = manager.run_bertopic(df)

    if cfg.llm.run_llm:
        manager.run_llm(df)

    df = manager.get_label_prediction(df)
    metrics = manager.run_validation(df, save=True)
