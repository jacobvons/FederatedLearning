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
