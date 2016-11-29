class DatasetLoader:
    def __init__(self, path):
        self.examples = []
        self.labels = []
        lines = self._read_lines(path)
        for line in lines:
            self.examples.append(list(map(float, line[0:-1])))
            self.labels.append(line[-1])

    def _read_lines(self, path):
        lines = []
        for line in open(path):
            lines.append(line.rstrip('\n').split(','))
        return lines
