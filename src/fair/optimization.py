import numpy as np
import scipy

from fair.agent import BaseAgent


class IntegerLinearProgram:
    def __init__(self, agents: list[BaseAgent]):
        """
        Args:
            agents (list[BaseAgent]): Agents whose constraints define the program
        """
        self.agents = agents
        self.A = None
        self.b = None

    def compile(self):
        """Create a single (block) constraint matrix for all agents

        Resulting block matrix A acts on an allocation vector that results from
        concatenating all allocation indicator vectors across all agents.

        Returns:
            IntegerLinearProgram: compiled IntegerLinearProgram
        """
        if self.A is not None and self.b is not None:
            return

        A_blocks = []
        bs = []
        for i, agent in enumerate(self.agents):
            valuation = agent.valuation.compile()
            A_block = [None] * len(self.agents)
            A_block[i] = valuation.constraints[0].to_sparse().A
            A_blocks.append(A_block)
            bs.append(valuation.constraints[0].to_sparse().b)

        self.A = scipy.sparse.bmat(A_blocks, format="csr")
        self.b = scipy.sparse.vstack(bs)

        return self

    def convert_allocation(self, X: type[np.ndarray | scipy.sparse.sparray]):
        """Convert an allocation matrix to a form that can be tested against constraints

        Args:
            X (type[np.ndarray  |  scipy.sparse.sparray]): Allocation matrix

        Raises:
            IndexError: There must exist at least one column in X for each agent

        Returns:
            scipy.sparse.csr_matrix: stacked allocation vector
        """
        if X.shape[1] < len(self.agents):
            raise IndexError(
                f"columns in allocation matrix: {X.shape[1]} cannot be less than agents: {len(self.agents)}"
            )

        return scipy.sparse.vstack(
            [scipy.sparse.csr_matrix(X[:, i]).T for i in range(len(self.agents))]
        )
