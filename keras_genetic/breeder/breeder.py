class Breeder:
    """Breeder is responsible for creating offspring."""

    def offspring(self, mother, father):
        """`offspring()` creates a new offspring from a mother and father model.

        Args:
            mother: keras_genetic.Individual.
            father: keras_genetic.Individual.
        Returns:
            a new offspring set of weights.
        """
        raise NotImplementedError(
            "`offspring()` must be implemented on the breeder class."
        )
