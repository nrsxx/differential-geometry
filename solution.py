import numpy as np
import math
import draw


class Mesh:
    def __init__(self, faces, coordinates=None):
        self.faces = faces
        vertices = set(i for f in faces for i in f)
        self.n = max(vertices) + 1
        if coordinates is not None:
            self.coordinates = np.array(coordinates)

        assert set(range(self.n)) == vertices
        for f in faces:
            assert len(f) == 3
        if coordinates is not None:
            assert self.n == len(coordinates)
            for c in coordinates:
                assert len(c) == 3

    @classmethod
    def fromobj(cls, filename):
        faces, vertices = draw.obj_read(filename)
        return cls(faces, vertices)

    def draw(self):
        draw.draw(self.faces, self.coordinates.tolist())

    def angleDefect(self, vertex):  # vertex is an integer (vertex index from 0 to self.n-1)

        sum_of_angles = 0
        for face in self.faces:
            if vertex in face:
                cur_vertices = [self.coordinates[vertex]]
                for v in face:
                    if v != vertex:
                        v = self.coordinates[v]
                        cur_vertices.append(v)
                lengths = np.zeros(0)
                for i in range(3):
                    lengths = np.append(lengths, [math.sqrt(np.sum((cur_vertices[(i+1) % 3]
                                                                    - cur_vertices[(i+2) % 3]) ** 2))])
                cur_angle = math.acos((lengths[1]*lengths[1] + lengths[2] * lengths[2]
                                       - lengths[0] * lengths[0])/(2.0 * lengths[1] * lengths[2]))
                sum_of_angles += cur_angle
        return math.pi * 2 - sum_of_angles

    def buildLaplacianOperator(self, anchors=None, anchor_weight=1.):
        # anchors is a list of vertex indices, anchor_weight is a positive number
        if anchors is None:
            anchors = []

        A = np.zeros((self.n, self.n))
        B = np.zeros((self.n, self.n))

        for f in self.faces:
            for i in range(3):
                a, b = self.coordinates[f[(i + 1) % 3]] - self.coordinates[f[i]], self.coordinates[f[(i + 2) % 3]] \
                       - self.coordinates[f[i]]
                alpha = math.acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                A[f[(i + 1) % 3]][f[(i + 2) % 3]] += 1 / (2 * math.tan(alpha))
                A[f[(i + 2) % 3]][f[(i + 1) % 3]] += 1 / (2 * math.tan(alpha))

        for i in range(self.n):
            B[i][i] += -np.sum(A[i])

        old_laplasian = A + B

        for i in range(self.n):
            old_laplasian[i] /= -old_laplasian[i][i]

        laplasian_add = np.zeros((len(anchors), self.n))
        for v in anchors:
            laplasian_add[anchors.index(v)][v] = anchor_weight

        laplasian = np.append(old_laplasian, laplasian_add, axis=0)

        return laplasian

    def smoothen(self):
        L = self.buildLaplacianOperator()
        self.coordinates += np.dot(L, self.coordinates)

    def transform(self, anchors, anchor_coordinates, anchor_weight=1.):
        # anchors is a list of vertex indices, anchor_coordinates is a list of same length of vertex coordinates
        # (arrays of length 3), anchor_weight is a positive number

        L = self.buildLaplacianOperator(anchors, anchor_weight)
        L_start = L[:self.n]
        delta = np.dot(L_start, self.coordinates)
        delta = np.append(delta, anchor_coordinates, axis=0)

        self.coordinates = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(L), L)), np.transpose(L)), delta)


def dragon():
    #mesh = Mesh.fromobj("dragon.obj")
    #mesh.smoothen()
    #mesh.draw()
    raise NotImplementedError
