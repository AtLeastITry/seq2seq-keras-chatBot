class OutputData:
    def __init__(self):
        self.num_lines = 0
        self.chars = set()
        self.num_tokens = 0
        self.max_len = 0

    def loadJSON(self, data):
        self.num_lines = data['num_lines']
        self.chars = data['chars']
        self.num_tokens = data['num_tokens']
        self.max_len = data['max_len']