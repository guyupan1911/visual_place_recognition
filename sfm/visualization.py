import sys

import pypangolin as pango
from OpenGL.GL import *

import numpy as np

def DrawCoordinate(pose):
    glPushMatrix()
    glMultTransposeMatrixd(pose)

    glLineWidth(3.0)
    glBegin(GL_LINES)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3d(0, 0, 0)
    glVertex3d(1, 0, 0)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3d(0, 0, 0)
    glVertex3d(0, 1, 0)
    glColor3f(0.0, 0.0, 1.0)
    glVertex3d(0, 0, 0)
    glVertex3d(0, 0, 1)
    glEnd()

    glPopMatrix()

def DrawCamera(pose):
    glPushMatrix()
    glMultTransposeMatrixd(pose)

    w = 0.06
    h = w * 0.75
    z = w * 0.6

    glColor3f(0.2, 1.0, 0.1)
    glLineWidth(2)

    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(w, h, z)
    glVertex3f(0, 0, 0)
    glVertex3f(w, -h, z)
    glVertex3f(0, 0, 0)
    glVertex3f(-w, -h, z)
    glVertex3f(0, 0, 0)
    glVertex3f(-w, h, z)
    glVertex3f(w, h, z)
    glVertex3f(w, -h, z)
    glVertex3f(-w, h, z)
    glVertex3f(-w, -h, z)
    glVertex3f(-w, h, z)
    glVertex3f(w, h, z)
    glVertex3f(-w, -h, z)
    glVertex3f(w, -h, z)
    glEnd()

    glBegin(GL_LINES)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3d(0, 0, 0)
    glVertex3d(w, 0, 0)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3d(0, 0, 0)
    glVertex3d(0, w, 0)
    glColor3f(0.0, 0.0, 1.0)
    glVertex3d(0, 0, 0)
    glVertex3d(0, 0, w)
    glEnd()

    glPopMatrix()


def DrawCloud(points):
    glColor3f(0,0,0)
    glPointSize(4)
    glBegin(GL_POINTS)
    for i in range(points.shape[1]):
        if points[3,i] < 0:
            continue
        glVertex3d(points[0,i], points[1,i], points[2,i])
    glEnd()