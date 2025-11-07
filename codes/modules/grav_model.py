# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:41:12 2021

@author: zampa
"""


# -----------------------------------------------------------------------------
# Import libraries

from . import utils as utl

# -----------------------------------------------------------------------------
# Set the aliases for some libraries from the utils and other module

np = utl.np
os = utl.os
sys = utl.sys
ogr = utl.ogr
osr = utl.osr
gdal = utl.gdal
plt = utl.plt
time = utl.time
copy = utl.copy

# Costante Francesco P.
# G = 6.67299 * 1e-11 # [m3/(kg *s^2)]

# Costante Uieda 
G = 6.6743e-11

# -----------------------------------------------------------------------------
def xyz2prisms( x, y, z, bottom=0, step=None, size=None, plot=False, grid=True,
                method="nearest" ) :
    
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    z = np.array(z).ravel()
    
    if type( z ) in ( int, float ) :
        z = np.repeat( z, len(x) )
    
    X, Y, step = utl.xy2XY( x, y, step=step, grid=grid, return_step=True )
    
    xp = X.ravel()
    yp = Y.ravel()
    
    if np.size( X.ravel() ) != np.size( x ) :
        Z,_ = utl.xyz2xy( ( x, y, z ), ( xp, yp ), method=method )
        zp = Z.ravel()
    else :
        zp = z
    
    if type( bottom ) in ( int, float ) :
        bottom = np.repeat( bottom, np.size(xp) )

    eb = np.array( xp ) - step[0]/2
    wb = np.array( xp ) + step[0]/2
    sb = np.array( yp ) - step[1]/2
    nb = np.array( yp ) + step[1]/2
    lb = np.array( bottom )
    ub = np.array( zp )

    prisms = np.column_stack( ( eb, wb, sb, nb, lb, ub ) )
    
    if plot is True :
        
        plt.figure() 
        plt.scatter( eb, sb, s=size, c='r', marker='+')
        plt.scatter( eb, nb, s=size, c='r', marker='+')
        plt.scatter( wb, sb, s=size, c='r', marker='+')
        plt.scatter( wb, nb, s=size, c='r', marker='+')
        plt.scatter( xp, yp, s=size, c=zp )
    
    return prisms

# -----------------------------------------------------------------------------   
def nagy_fz( x, y, z ):

    r = ( x**2 + y**2 + z**2 )**( 1/2 )  
    
    div = np.nan_to_num( ( x * y ) / ( z * r )  ) 
    logyr = np.nan_to_num( np.log( y + r ) )  
    logxr = np.nan_to_num( np.log( x + r ) ) 
    # ---

    fzi = x * logyr + y * logxr - ( z * np.arctan( div ) )

    return fzi

# -----------------------------------------------------------------------------  
def nagy_gz( array ):
    
    x0, y0, z0 = array[:,0], array[:,1], array[:,2]
    
    x1, x2, y1, y2, z1, z2 = array[:,3]-x0, array[:,4]-x0, \
                             array[:,5]-y0, array[:,6]-y0, \
                             array[:,7]-z0, array[:,8]-z0 
    
    rho = array[:,9]                         
    
    gzi = ( ( ( nagy_fz( x2, y2, z2 ) - nagy_fz( x1, y2, z2 ) )
             -( nagy_fz( x2, y1, z2 ) - nagy_fz( x1, y1, z2 ) ) )
           -( ( nagy_fz( x2, y2, z1 ) - nagy_fz( x1, y2, z1 ) )
             -( nagy_fz( x2, y1, z1 ) - nagy_fz( x1, y1, z1 ) ) ) )
    
    gzi = G * rho * gzi *1e5
    
    
    return gzi


# -----------------------------------------------------------------------------  
def prism_to_faces(prism):
    """
    Convert a prism defined as [west, east, south, north, bottom, top]
    into a list of polygonal faces (each face is a list of 3D vertices).

    Parameters
    ----------
    prism : array-like of shape (6,)
        The bounds of the prism: [x_min, x_max, y_min, y_max, z_min, z_max]

    Returns
    -------
    faces : list of faces
        Each face is a list of 3D points (numpy arrays)
    """
    xw, xe, ys, yn, zb, zu = prism

    # Define all 8 vertices (bottom first, then top)
    v = [
        np.array([xw, ys, zb]),  # 0 bottom-southwest
        np.array([xe, ys, zb]),  # 1 bottom-southeast
        np.array([xe, yn, zb]),  # 2 bottom-northeast
        np.array([xw, yn, zb]),  # 3 bottom-northwest
        np.array([xw, ys, zu]),  # 4 top-southwest
        np.array([xe, ys, zu]),  # 5 top-southeast
        np.array([xe, yn, zu]),  # 6 top-northeast
        np.array([xw, yn, zu]),  # 7 top-northwest
    ]

    # Define faces (counter-clockwise as seen from outside)
    faces = [
        [v[0], v[1], v[2], v[3]],  # bottom
        [v[4], v[5], v[6], v[7]],  # top
        [v[0], v[1], v[5], v[4]],  # south
        [v[1], v[2], v[6], v[5]],  # east
        [v[2], v[3], v[7], v[6]],  # north
        [v[3], v[0], v[4], v[7]],  # west
    ]

    return faces

# -----------------------------------------------------------
def unit_normal(face):
    p0 = face[0]
    n = np.zeros(3)
    for i in range(1, len(face) - 1):
        v1 = face[i] - p0
        v2 = face[i + 1] - p0
        n += np.cross(v2, v1)
    return n / (np.linalg.norm(n) + 1e-12)

def anglegravi(p1, p2, p3, Un, epsilon=1e-6):
    p1 = p1 + np.random.uniform(-epsilon, epsilon, size=3)
    p2 = p2 + np.random.uniform(-epsilon, epsilon, size=3)
    p3 = p3 + np.random.uniform(-epsilon, epsilon, size=3)

    inout = np.sign(np.sum(Un * p1))
    if inout == 0:
        return 0.0, 1

    if inout > 0:
        x1, y1, z1 = p3
        x2, y2, z2 = p2
        x3, y3, z3 = p1
    else:
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        x3, y3, z3 = p3

    n1 = np.array([y2*z1 - y1*z2, x1*z2 - x2*z1, x2*y1 - x1*y2], dtype=float)
    n2 = -np.array([y3*z2 - y2*z3, x2*z3 - x3*z2, x3*y2 - x2*y3], dtype=float)

    n1 /= np.linalg.norm(n1) + 1e-12
    n2 /= np.linalg.norm(n2) + 1e-12

    perp = np.sign(np.dot([x3, y3, z3], n1))
    ang = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))

    if perp < 0:
        ang = 2 * np.pi - ang

    return ang, perp

# -----------------------------------------------------------
def solid_angle(polygon, obs):
    polygon = [np.array(v, dtype=float) - obs for v in polygon]
    normal = unit_normal(np.array(polygon))
    omega = 0.0
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i+1) % n]
        p3 = polygon[(i+2) % n]
        ang, _ = anglegravi(p1, p2, p3, normal)
        omega += ang
    return omega - (n - 2) * np.pi

# ---
def prism_to_corners_faces(prism):
    """
    Converts a rectangular prism [x_min, x_max, y_min, y_max, z_min, z_max]
    into vertices (corners) and faces (as index lists).

    Parameters
    ----------
    prism : array-like of shape (6,)
        The prism bounds: [xw, xe, ys, yn, zb, zu]

    Returns
    -------
    corners : ndarray of shape (8, 3)
        Coordinates of the 8 prism vertices.

    faces : list of lists
        Each face is a list in the format [n, i1, i2, ..., in]
        following MATLAB convention: n = number of vertices,
        i1...in are indices into corners (0-based).
    """
    xw, xe, ys, yn, zb, zu = prism

    corners = np.array([
        [xw, ys, zb],  # 0: bottom-southwest
        [xe, ys, zb],  # 1: bottom-southeast
        [xe, yn, zb],  # 2: bottom-northeast
        [xw, yn, zb],  # 3: bottom-northwest
        [xw, ys, zu],  # 4: top-southwest
        [xe, ys, zu],  # 5: top-southeast
        [xe, yn, zu],  # 6: top-northeast
        [xw, yn, zu],  # 7: top-northwest
    ])

    faces = [
        [4, 0, 1, 2, 3],  # bottom
        [4, 4, 5, 6, 7],  # top
        [4, 0, 1, 5, 4],  # south
        [4, 1, 2, 6, 5],  # east
        [4, 2, 3, 7, 6],  # north
        [4, 3, 0, 4, 7],  # west
    ]

    return corners, faces

# =============================================================================
class nagy() :
    
    def __init__( self, coordinates, prism, density ) :
        
        self.coordinates = coordinates
        self.prism = prism
        self.density = density
        
        if type( prism ) in ( list, tuple ) :
            self.prism = np.asarray( prism )   
     
        if type( coordinates ) in ( list, tuple ) :
            self.coordinates = np.array( coordinates, ndmin=2 )
            
        if np.size( density ) == 1 :
            self.density = np.repeat( density, self.prism.shape[0] )
            
        if type( density ) in ( list, tuple ) :
            self.density = np.asarray( density )
            
    # ------------------------------------------------------------------------- 
    def gz( self ) :
        
        np.seterr(divide='ignore', invalid='ignore')
        
        self.pgz = np.zeros( self.coordinates.shape[0] )
        
        for i, c in enumerate( self.coordinates ) :
            
            C = np.tile( self.coordinates[i,:], ( self.prism.shape[0], 1 ) )
            
            array = np.column_stack( ( C, self.prism, self.density ) ) 
            
            # self.pgz[i] = np.sum( np.apply_along_axis( nagy_gz, 1, array ) )
            self.pgz[i] = np.sum( nagy_gz(array) )
            
        return self.pgz 
    
# =============================================================================
class PolyGrav:
    def __init__(self, corners, faces, density):
        self.corners = np.array(corners)
        self.faces = faces
        self.density = density

    def compute_gz(self, obs):
        cor = self.corners - obs
        gz = 0.0
        for face in self.faces:
            nsides = face[0]
            indices = face[1:1+nsides]
            verts = [cor[i] for i in indices]
            corner_face = [self.corners[i] for i in indices]

            if len(corner_face) < 3:
                continue

            un = unit_normal(np.array(corner_face))
            fsign = np.sign(np.dot(un, verts[0]))
            dp1 = np.dot(un, verts[0])

            # Solid angle
            W = 0.0
            for i in range(nsides):
                p1 = verts[i % nsides]
                p2 = verts[(i + 1) % nsides]
                p3 = verts[(i + 2) % nsides]
                ang, _ = anglegravi(p1, p2, p3, un)
                W += ang
            omega = -fsign * (W - (nsides - 2) * np.pi)

            # Line integrals (faithful to MATLAB)
            PQR = np.zeros(3)
            for i in range(nsides):
                p1 = verts[i % nsides]
                p2 = verts[(i + 1) % nsides]
                V = p2 - p1
                L = np.linalg.norm(V)
                if L < 1e-12:
                    continue
                b = 2 * np.dot(V, p1)
                r1 = np.linalg.norm(p1)
                b2 = b / (2 * L)
                denom = r1 + b2

                if denom == 0:
                    I = 0.0
                else:
                    num = np.sqrt(L**2 + b + r1**2) + L + b2
                    try:
                        I = (1 / L) * np.log(num / denom)
                    except:
                        I = 0.0
                if r1 + b2 == 0:
                    V = -V
                    b = 2 * np.dot(V, p1)
                    b2 = b / (2 * L + 1e-12)
                if r1 + b2 != 0:
                    try:
                        I = (1 / L) * np.log((np.sqrt(L**2 + b + r1**2) + L + b2) / (r1 + b2))
                    except:
                        I = 0.0
                    PQR += I * V

            l, m, n = un
            p, q, r = PQR

            gz += -self.density * G * dp1 * (n * omega + m * p - l * q)

        return gz

# =============================================================================
class GotzeGrav:
    def __init__(self, corners, faces, density):
        self.corners = np.array(corners)
        self.faces = faces
        self.density = density

    def compute_gz(self, obs):
        cor = self.corners - obs
        gz = 0.0
        for face in self.faces:
            nsides = face[0]
            indices = face[1:1+nsides]
            verts = [cor[i] for i in indices]
            corner_face = [self.corners[i] for i in indices]

            if len(corner_face) < 3:
                continue

            un = unit_normal(np.array(corner_face))
            fsign = np.sign(np.dot(un, verts[0]))
            dp1 = np.dot(un, verts[0])

            W = 0.0
            for i in range(nsides):
                p1 = verts[i % nsides]
                p2 = verts[(i + 1) % nsides]
                p3 = verts[(i + 2) % nsides]
                ang, _ = anglegravi(p1, p2, p3, un)
                W += ang
            omega = -fsign * (W - (nsides - 2) * np.pi)

            PQR = np.zeros(3)
            for i in range(nsides):
                p1 = verts[i % nsides]
                p2 = verts[(i + 1) % nsides]
                V = p2 - p1
                L = np.linalg.norm(V)
                if L < 1e-12:
                    continue
                b = 2 * np.dot(V, p1)
                r1 = np.linalg.norm(p1)
                b2 = b / (2 * L + 1e-12)
                if r1 + b2 == 0:
                    V = -V
                    b = 2 * np.dot(V, p1)
                    b2 = b / (2 * L + 1e-12)
                if r1 + b2 != 0:
                    try:
                        I = (1 / L) * np.log((np.sqrt(L**2 + b + r1**2) + L + b2) / (r1 + b2))
                    except:
                        I = 0.0
                    PQR += I * V

            l, m, n = un
            p, q, r = PQR

            gz += -self.density * G * dp1 * (n * omega + m * p - l * q)

        return gz * 1e5  # convert from m/s^2 to mGal


def build_edges_and_normals(corners, faces):
    """
    Ricrea la struttura Edge e calcola le normali Un per ogni faccia,
    in modo fedele al comportamento di gravicalc2.m

    Parameters
    ----------
    corners : ndarray, shape (N_vertices, 3)
        Coordinate dei vertici del poliedro.
    faces : list of lists
        Ogni faccia ha il formato [n, i1, i2, ..., in] con indici 0-based.

    Returns
    -------
    edges : ndarray, shape (N_edges, 8)
        Matrice con [Vx, Vy, Vz, L, I, done, idx1, idx2]
        (per ora I e done non vengono usati, ma sono presenti per compatibilità).
    normals : ndarray, shape (N_faces, 3)
        Normale unitaria per ogni faccia.
    """
    n_faces = len(faces)
    n_edges = sum(f[0] for f in faces)
    edges = np.zeros((n_edges, 8))
    edge_count = 0

    for f_idx, face in enumerate(faces):
        n = face[0]
        idx = face[1:]
        idx_loop = idx + [idx[0]]  # chiusura della faccia

        for i in range(n):
            ends = [idx_loop[i], idx_loop[i+1]]
            p1 = corners[ends[0]]
            p2 = corners[ends[1]]
            V = p2 - p1
            L = np.linalg.norm(V)
            edges[edge_count, 0:3] = V
            edges[edge_count, 3] = L
            edges[edge_count, 6:8] = ends
            edge_count += 1

    normals = np.zeros((n_faces, 3))
    for f_idx, face in enumerate(faces):
        n = face[0]
        idx = face[1:]
        ss = np.zeros(3)
        for i in range(1, n - 1):
            v1 = corners[idx[i+1]] - corners[idx[0]]
            v2 = corners[idx[i]]   - corners[idx[0]]
            ss += np.cross(v2, v1)
        norm = np.linalg.norm(ss)
        if norm != 0:
            normals[f_idx] = ss / norm
        else:
            normals[f_idx] = ss  # vettore nullo se degenerata

    return edges, normals


def anglegravi_exact(p1, p2, p3, Un):
    """
    Traduzione fedele della funzione MATLAB 'anglegravi'.

    Parametri
    ---------
    p1, p2, p3 : ndarray
        Vertici del triangolo (3,) spostati rispetto al punto di osservazione (origine).
    Un : ndarray
        Normale unitaria uscente della faccia.

    Ritorna
    -------
    ang : float
        Angolo tra i piani (in radianti).
    perp : int
        +1 se il triangolo è in senso antiorario rispetto a Un, -1 se orario.
    """
    inout = np.sign(np.dot(Un, p1))
    
    if inout == 0:
        return 0.0, 1

    # Coordinate interscambiate come da codice MATLAB
    if inout > 0:
        x1, y1, z1 = p3
        x2, y2, z2 = p2
        x3, y3, z3 = p1
    else:
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        x3, y3, z3 = p3

    n1 = np.array([
        y2 * z1 - y1 * z2,
        x1 * z2 - x2 * z1,
        x2 * y1 - x1 * y2
    ], dtype=float)

    n2 = -np.array([
        y3 * z2 - y2 * z3,
        x2 * z3 - x3 * z2,
        x3 * y2 - x2 * y3
    ], dtype=float)

    n1 /= np.linalg.norm(n1) + 1e-12
    n2 /= np.linalg.norm(n2) + 1e-12

    perp = np.sign(np.dot(np.array([x3, y3, z3]), n1))
    ang = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))

    if perp < 0:
        ang = 2 * np.pi - ang

    return ang, perp

def build_edge_table(corners, faces):
    """
    Costruisce una tabella degli spigoli con caching e indicizzazione simmetrica,
    equivalente a Edge in gravicalc2.m.

    Returns
    -------
    edge_table : dict
        Chiavi (i,j) e (j,i), valori dict con V, L, I, done.
    edge_list : list
        Lista completa degli edge (ordine per faccia).
    """
    edge_table = {}
    edge_list = []

    for f_idx, face in enumerate(faces):
        n = face[0]
        idx = face[1:]
        idx_loop = idx + [idx[0]]

        for i in range(n):
            a, b = idx_loop[i], idx_loop[i + 1]
            key = (a, b)
            rev_key = (b, a)
            if key not in edge_table and rev_key not in edge_table:
                p1 = corners[a]
                p2 = corners[b]
                V = p2 - p1
                L = np.linalg.norm(V)
                edge_table[key] = {
                    'V': V,
                    'L': L,
                    'I': 0.0,
                    'done': False,
                    'verts': (a, b)
                }
            edge_list.append((a, b))

    return edge_table, edge_list

def gravicalc2_full(obs_points, corners, faces, density):
    """
    Riscrittura fedele di gravicalc2.m in Python con edge caching, simmetria e struttura Edge.
    """
    G = 6.6732e-11 * 1e5  # in mGal
    obs_points = np.atleast_2d(obs_points)
    N_obs = obs_points.shape[0]

    normals = np.zeros((len(faces), 3))
    for f_idx, face in enumerate(faces):
        n = face[0]
        idx = face[1:]
        ss = np.zeros(3)
        for i in range(1, n - 1):
            v1 = corners[idx[i + 1]] - corners[idx[0]]
            v2 = corners[idx[i]] - corners[idx[0]]
            ss += np.cross(v2, v1)
        norm = np.linalg.norm(ss)
        normals[f_idx] = ss / norm if norm != 0 else ss

    edge_table, edge_list = build_edge_table(corners, faces)

    Gx = np.zeros(N_obs)
    Gy = np.zeros(N_obs)
    Gz = np.zeros(N_obs)

    for pr in range(N_obs):
        gx = gy = gz = 0.0
        obs = obs_points[pr]
        cor = corners - obs

        edge_index = 0
        for f_idx, face in enumerate(faces):
            nsides = face[0]
            idx = face[1:]
            indx = idx + [idx[0], idx[1]]
            crs = [cor[i] for i in idx]
            un = normals[f_idx]
            fsign = np.sign(np.dot(un, crs[0]))
            dp1 = np.dot(crs[0], un)
            dp = abs(dp1)

            if dp == 0:
                omega = 0.0
            else:
                W = 0.0
                for i in range(nsides):
                    p1 = cor[indx[i]]
                    p2 = cor[indx[i+1]]
                    p3 = cor[indx[i+2]]
                    ang, _ = anglegravi_exact(p1, p2, p3, un)
                    W += ang
                omega = -fsign * (W - (nsides - 2) * np.pi)

            PQR = np.zeros(3)
            for i in range(nsides):
                a = indx[i]
                b = indx[i + 1]
                key = (a, b)
                rev_key = (b, a)

                p1 = cor[a]
                p2 = cor[b]
                V = p2 - p1
                L = np.linalg.norm(V)

                if L < 1e-12:
                    continue

                if key in edge_table:
                    edge = edge_table[key]
                else:
                    edge = edge_table[rev_key]
                    V = -edge['V']  # verso opposto

                if edge['done']:
                    I = edge['I']
                else:
                    chsgn = 1
                    dot_prod = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
                    if np.isclose(dot_prod, 1.0, atol=1e-12) and np.linalg.norm(p1) > np.linalg.norm(p2):
                        chsgn = -1
                        p1, p2 = p2, p1
                        V = -V

                    b_val = 2 * np.dot(V, p1)
                    r1 = np.linalg.norm(p1)
                    b2 = b_val / (2 * L)
                    denom = r1 + b2

                    if denom == 0:
                        V = -V
                        b_val = 2 * np.dot(V, p1)
                        b2 = b_val / (2 * L)

                    if r1 + b2 != 0:
                        I = (1 / L) * np.log((np.sqrt(L**2 + b_val + r1**2) + L + b2) / (r1 + b2))
                    else:
                        I = 0.0

                    I *= chsgn
                    edge['I'] = I
                    edge['done'] = True
                    # Salva anche nel verso opposto
                    if rev_key in edge_table:
                        edge_table[rev_key]['I'] = I
                        edge_table[rev_key]['done'] = True

                PQR += I * V

            l, m, n = un
            p, q, r = PQR

            if dp != 0:
                gx += -density * G * dp1 * (l * omega + n * q - m * r)
                gy += -density * G * dp1 * (m * omega + l * r - n * p)
                gz += -density * G * dp1 * (n * omega + m * p - l * q)

            edge_index += nsides

        Gx[pr] = gx
        Gy[pr] = gy
        Gz[pr] = gz

    return Gx, Gy, Gz







