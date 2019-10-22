"""
Module to load object
"""


class OBJ:
    """
    Class to load object
    """

    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                verts = list(map(float, values[1:4]))
                if swapyz:
                    verts = verts[0], verts[2], verts[1]
                self.vertices.append(verts)
            elif values[0] == 'vn':
                verts = list(map(float, values[1:4]))
                if swapyz:
                    verts = verts[0], verts[2], verts[1]
                self.normals.append(verts)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            # elif values[0] in ('usemtl', 'usemat'):
                #material = values[1]
            # elif values[0] == 'mtllib':
                #self.mtl = MTL(values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for verts in values[1:]:
                    w_val = verts.split('/')
                    face.append(int(w_val[0]))
                    if len(w_val) >= 2 and len(w_val[1]) > 0:
                        texcoords.append(int(w_val[1]))
                    else:
                        texcoords.append(0)
                    if len(w_val) >= 3 and len(w_val[2]) > 0:
                        norms.append(int(w_val[2]))
                    else:
                        norms.append(0)
                #self.faces.append((face, norms, texcoords, material))
                self.faces.append((face, norms, texcoords))
