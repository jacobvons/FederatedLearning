import re
import itertools


class ArgReader:

    def __init__(self, file_path: str):
        self.args = []
        self.path = file_path

    def parse(self):
        with open(self.path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                args = {}
                raw = line.strip().split(",")
                for arg in raw:
                    if arg.startswith("#"):
                        continue
                    name, value = arg.split(":")
                    args[name.strip()] = value.strip()
                self.args.append(args)


class HyperParamGenerator:

    def __init__(self, file_path):
        self.file_path = file_path
        self.hps = {}
        self.configs = []

    def read(self):
        with open(self.file_path, "r") as f:
            for line in f:
                if line.startswith("#") or line == "\n":
                    continue
                hp, values = [re.sub(" ", "", i) for i in line.split(":")]
                values = [float(i) for i in values.split(",")]
                self.hps[hp] = values

    def config_gen(self):
        self.read()
        vals = [v for v in self.hps.values()]
        combinations = list(itertools.product(*vals))
        self.configs = [dict(zip(self.hps.keys(), v)) for v in combinations]


if __name__ == "__main__":
    a = HyperParamGenerator("./hyper_configs.csv")
    a.config_gen()
