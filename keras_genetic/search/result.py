class SearchResult:
    """SearchResult contains the results for a search call."""

    def __init__(self, results):
        self.results = results

    @property
    def best(self):
        return self.results[0]
