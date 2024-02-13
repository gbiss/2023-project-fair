import numpy as np
import scipy

from fair.agent import BaseAgent
from fair.item import ScheduleItem


class IntegerLinearProgram:
    def __init__(self, agents: list[BaseAgent]):
        """
        Args:
            agents (list[BaseAgent]): Agents whose constraints define the program
        """
        self.agents = agents
        self.A = None
        self.b = None
        self.c = None
        self.bounds = None
        self.constraint = None

    def compile(self):
        """Create a single (block) constraint matrix for all agents

        Resulting block matrix A acts on an allocation vector that results from
        concatenating all allocation indicator vectors across all agents.

        Returns:
            IntegerLinearProgram: compiled IntegerLinearProgram
        """
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

    def add_constraint(self, Ap, bp):
        """Augment constraints beyond those supplied by the agent

        Args:
            Ap (np.ndarray | scipy.sparse.matrix): Constraint matrix
            bp (np.ndarray | scipy.sparse.matrix): Limit vector

        Raises:
            AttributeError: Constraints cannot be formulated until the ILP is compiled
            AttributeError: Shapes of Ap and bp must be compatible
            AttributeError: Shapes of self.A and Ap must be compatible

        Returns:
            IntegerLinearProgram: ILP with augmented constraints
        """
        if self.A is None or self.b is None:
            raise AttributeError("IntegerLinearProgram must be compiled first")

        if Ap.shape[0] != bp.shape[0]:
            raise AttributeError("rows in Ap must match rows in bp")

        if Ap.shape[1] != self.A.shape[1]:
            raise AttributeError("columns in Ap must match columns in A")

        self.A = scipy.sparse.vstack([self.A, scipy.sparse.csr_matrix(Ap)])
        self.b = scipy.sparse.vstack([self.b, scipy.sparse.csr_matrix(bp)])

        return self

    def formulateUSW(self):
        """Put previously compiled constraints into scipy optimization format

        Raises:
            AttributeError: ILP cannot be formulated until it is compiled

        Returns:
            IntegerLinearProgram: self
        """
        if self.A is None or self.b is None:
            raise AttributeError("IntegerLinearProgram must be compiled first")

        n, m = self.A.shape
        self.c = -np.ones((m,))
        self.bounds = scipy.optimize.Bounds(0, 1)
        self.constraint = scipy.optimize.LinearConstraint(
            self.A, ub=self.b.toarray().reshape((n,))
        )

        return self

    def solve(self):
        """Solve using scipy.optimize.milp (Mixed Integer Linear Programming)

        Raises:
            ValueError: Thrown if no optimal solutuion was found

        Returns:
            np.ndarray: optimal allocation
        """
        res = scipy.optimize.milp(
            c=self.c, integrality=1, bounds=self.bounds, constraints=self.constraint
        )

        if not res.success:
            raise ValueError("no optimal solution found")

        return res.x

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


class StudentAllocationProgram(IntegerLinearProgram):
    """An Integer Linear Program for allocating courses to students"""

    def __init__(self, students: list[BaseAgent], schedule: list[ScheduleItem]):
        """
        Args:
            students (list[BaseAgent]): Students whose constraints define the program
            schedule (list[ScheduleItem]): Items from which student preferences are constructed
        """
        self.schedule = schedule
        super().__init__(students)

    def compile(self):
        """Create a single (block) constraint matrix for all students

        Resulting block matrix A acts on an allocation vector that results from
        concatenating all allocation indicator vectors across all students. Beyond
        student linear constraints, also add capacity constraints for all courses
        in schedule.

        Returns:
            StudentAllocationProgram: compiled StudentAllocationProgram
        """
        super().compile()

        columns = self.A.shape[1]
        A = scipy.sparse.lil_matrix((len(self.schedule), columns), dtype=np.int64)
        b = scipy.sparse.lil_matrix((len(self.schedule), 1), dtype=np.int64)
        extents = [
            student.valuation.compile().constraints[0].extent for student in self.agents
        ]
        for row, item in enumerate(self.schedule):
            block_offset = 0
            for i in range(len(self.agents)):
                block_idx = block_offset + item.index
                A[row, block_idx] = 1
                block_offset += extents[i]
            b[row, 0] = item.capacity

        self.add_constraint(A.tocsr(), b.tocsr())

        return self
