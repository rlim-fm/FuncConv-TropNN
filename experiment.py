from train import Processor
from visualization import Visualizer
from itertools import product

class Experiment:
    def __init__(self, base_config=None, ivs=None, *, trials=5):
        self.base_config = base_config or {}
        self.ivs = ivs or {}
        self._create_configs()
        self.results = []
        self.trials = trials

    def run_grid(self):
        for config in self.configs:
            for seed in range(self.trials):
                processor = Processor(**config)
                processor.run()
            self.results.append((config, processor.predict(), processor.loss_history))

    def _create_configs(self):
        iv_vals = self.ivs.values()
        iv_val_combs = product(*iv_vals)
        # Create a list[dict] of changed variables
        iv_combs = [{key: val for key, val in zip(self.ivs.keys(), val_comb)} for val_comb in iv_val_combs]
        self.configs = [self.base_config.copy() | comb for comb in iv_combs]

def main():
    pass

if __name__ == '__main__':
    main()
