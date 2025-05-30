# -*- coding: utf-8 -*-
"""
Created on Wed May 28 13:52:59 2025

This script contains a class for extracting the vectors attached to each X-Y
position in a 3-D datacube.

@author: agilj
"""

import sys
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

"""
The main extractor function holds the parameters for the display. When called,
it instantiates a overseer class that manages the display windows.
CALL ARGUMENTS:
    cubes: A tuple of 3-D ndarrays. The datacubes do not need to be the same 
           size, but if they are not, all datacubes will be interpolated onto 
           the grid defined by the first datacube. The first datacube in the
           tuple is the one that will be displayed. The other cubes will have
           the interpolated vectors associated with each X-Y point displayed
           upon mouseover.
    xaxis: A tuple of 1-D ndarrays. The x-axis corresponding to each datacube.
"""
class extractor():
    def __init__(self, numRows=6, rgbType='Si-incl'):
        self.numRows = numRows
        self.rgbType = rgbType
    
    def __call__(self, cubes, xaxis):
        app = QtWidgets.QApplication(sys.argv)
        self.windowManager = _overseer(numRows=self.numRows,
                                       rgbType=self.rgbType)
        self.windowManager(cubes,xaxis)
        sys.exit(app.exec_())
        
        del self.windowManager.cubes, self.windowManager.xaxis


"""
Class to manage the datacube display.
"""
class _overseer(QtWidgets.QMainWindow):
    sendData_currLoc_signal = QtCore.pyqtSignal(tuple)
    sendData_clicked_signal = QtCore.pyqtSignal(tuple)

    
    def __init__(self, numRows, rgbType):
        super().__init__()
        
        self.numRows = numRows
        self.rgbType = rgbType
        
    def __call__(self, cubes, xaxis):
        # Put all cubes on the same scale. (Uses first cube as reference)
        ref = cubes[0]
        cubesInterp = [ref]
        for cube in cubes[1:]:
            cubesInterp.append(interpCube(ref,cube))
        cubesInterp = tuple(cubesInterp)
        
        # Store these in the class to make passing data around easier.
        self.cubes = cubesInterp
        self.xaxis = xaxis
        
        # Make windows.
        self._setupGUI(cubesInterp, xaxis)
    
    def _setupGUI(self, cubes, xaxis):
        # Set title.
        self.setWindowTitle('Main Image')
        # Set width and height, then move to center of the screen.
        width = self.cubes[0].shape[0] # image size plus border.
        height = self.cubes[0].shape[1] # image size plus border.
        self.setGeometry(0, 0, width, height)
        self.move(QtWidgets.QDesktopWidget().availableGeometry().center() \
                  - self.frameGeometry().center())
        
        # Make an false-color RGB version of the datacube.
        region = np.ones((self.cubes[0].shape[2]),dtype=bool)
        self.rgb_forDisp = makeRGB(self.cubes[0],
                                   region,
                                   method='Si-incl',
                                   cubeX=xaxis)
        
        # Scale and normalize for display
        self.rgb_forDisp = (255*self.rgb_forDisp/self.rgb_forDisp.max())
        self.rgb_forDisp = self.rgb_forDisp.astype(np.uint8)
        
        # Now display the RGB image in a display window.
        self._makeDisplay()
        
        # Setup cursor tracking.
        self.tracker = mouseTracker(self.label)
        
        # Call classes that manage other windows.
            # Vector display.
        miny = np.inf
        maxy = -np.inf
        for cube in self.cubes:
            miny = np.minimum(cube.min(),miny)
            maxy = np.maximum(cube.max(),maxy)
        self.vecDisplay = vecDisplay(miny, maxy)
            # Vector archive.
        self.vecArchive = vecArchive(self.numRows)
        # Now connect slots and signals.
        self.tracker.posChanged_signal.connect(self.extractVector)
        self.tracker.clicked_signal.connect(self.saveVector)
        self.sendData_currLoc_signal.connect(self.vecDisplay.plotData)
        self.sendData_clicked_signal.connect(self.vecArchive.plotData)
    
        self.show()

    def _makeDisplay(self):
        # Make label that the image will be put into.
        self.label = QtWidgets.QLabel(self)
        # Convert the image into a Qt-readable format.
        image = QtGui.QImage(self.rgb_forDisp,
                             self.rgb_forDisp.shape[1],
                             self.rgb_forDisp.shape[0],
                             self.rgb_forDisp.shape[1] * 3,
                             QtGui.QImage.Format_RGB888)
        # Labels take pixmaps, so convert to that.
        pixmap = QtGui.QPixmap(image)
        # Put the pixmap into the label.
        self.label.setPixmap(pixmap)
        # Set picture as main thing in the window.
        self.label.setFixedSize(self.rgb_forDisp.shape[1],
                                self.rgb_forDisp.shape[0])
        self.setCentralWidget(self.label)
    
    
    @QtCore.pyqtSlot(QtCore.QPoint)
    def extractVector(self, pos):
        # Extract vector from datacube. Send it to vector display window.
        vecs = []
        for cube in self.cubes:
            vecs.append(cube[pos.y(),pos.x()])
        self.sendData_currLoc_signal.emit((self.xaxis,vecs))
        
    @QtCore.pyqtSlot(QtCore.QPoint)
    def saveVector(self, pos):
        # Extract vector from datacube. Save it in the archive.
        vecs = []
        for cube in self.cubes:
            vecs.append(cube[pos.y(),pos.x()])
        self.sendData_clicked_signal.emit((self.xaxis,vecs,(pos.x(),pos.y())))
        # Add circle to image to mark selected point.
        self.addCircle(pos, self.rgb_forDisp[pos.y(),pos.x(),:])
        
    def addCircle(self, pos, rgbVal):
        rgbVal = rgbVal.astype(float)
        # Get pixmap currently being used.
        pixmap = self.label.pixmap()
        # Make a painter to draw on the pixmap.
        qp = QtGui.QPainter(pixmap)
        # Compute the color brightness of the pixel.
        contrastVal = (rgbVal[0]*299 + rgbVal[1]*587 + rgbVal[2]*114)/1000
        # Depending on color brightness, use black or white.
        if contrastVal > 128:
            qc = QtCore.Qt.black
        else:
            qc = QtCore.Qt.white
        qp.setPen(QtGui.QPen(qc, 1))
        # Draw a circle at that pixel.
        qp.drawEllipse(QtCore.QPoint(pos.x(),pos.y()), # Center
                       2, # Semi-major axis
                       2)  # Semi-minor axis
        # Close the painter!
        qp.end()
        # Now use the modified pixmap.
        self.label.setPixmap(pixmap)

"""
Supporting class for mouse tracking. 
"""
class mouseTracker(QtCore.QObject):
    posChanged_signal = QtCore.pyqtSignal(QtCore.QPoint)
    clicked_signal = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, widget):
        super().__init__(widget)
        self._widget = widget
        self.widget.setMouseTracking(True)
        self.widget.installEventFilter(self)
        self.clickedDown = False

    @property
    def widget(self):
        return self._widget

    def eventFilter(self, o, e):
        if o is self.widget and e.type() == QtCore.QEvent.MouseMove:
            self.posChanged_signal.emit(e.pos())
            self.clickedDown = False
        elif o is self.widget and e.type() == QtCore.QEvent.MouseButtonPress:
            print('Down!')
            self.clickedDown = True
        elif o is self.widget and e.type() == QtCore.QEvent.MouseButtonRelease\
            and self.clickedDown:
            print('Up!')
            self.clicked_signal.emit(e.pos())
            self.clickedDown = False

        return super().eventFilter(o, e)

"""
Class to manage the vector display window.
"""
class vecDisplay(QtWidgets.QWidget):
    def __init__(self, miny, maxy, colors=None, numColors=10, seed=5):
        super().__init__()
        # Set y-axis bounds.
        self.miny = miny
        self.maxy = maxy
        # Get array of plotting colors.
        if colors==None:    
            self.numColors = numColors
            rng = np.random.default_rng(seed=seed)
            self.colors = rng.integers(0,255,(3,numColors))
        else:
            self.numColors = colors.shape[1]
            self.colors = colors
        # Make display window.
        self._setupWindow()
    
    def _setupWindow(self):
        self.setGeometry(0,0,100,100)
        self.plot = pg.plot(title='Vector Display')
        
    @QtCore.pyqtSlot(tuple)
    def plotData(self,data):
        x = data[0]
        yset = data[1]
        self.plot.plotItem.plot(x,yset.pop(0),clear=True, pen='w')
        for ind,y in enumerate(yset):
            cInd = np.mod(ind,self.numColors)
            self.plot.plotItem.plot(x,
                                    y,
                                    clear=False,
                                    pen=self.colors[:,cInd])
        self.plot.plotItem.setXRange(x.min(),
                                     x.max(),
                                     padding=0.05)
        self.plot.plotItem.setYRange(self.miny,
                                     self.maxy,
                                     padding=0.05)            

"""
Class to manage display of selected points in the image.
"""
class vecArchive(QtWidgets.QWidget):
    def __init__(self, numRows, colors=None, numColors=10, seed=5):
        super().__init__()
        # Store how many rows we want in the visualization.
        self.numRows = numRows
        self.numPlots = 0 # Used to track where new data should be stored.
        # Get array of plotting colors.
        if colors==None:    
            self.numColors = numColors
            rng = np.random.default_rng(seed=seed)
            self.colors = rng.integers(0,255,(3,numColors))
        else:
            self.numColors = colors.shape[1]
            self.colors = colors
        # For this display, we adaptively change the min and max values
        # according to the data we've selected.
        self.miny = np.inf
        self.maxy = -np.inf
        # Make display window.
        self._setupWindow()
        
    def _setupWindow(self):
        self.setGeometry(0,0,100,100)
        self.pglayout = pg.GraphicsLayoutWidget(title='Saved Vectors',
                                                show=True)
        
    @QtCore.pyqtSlot(tuple)
    def plotData(self,data):
        # First get the new min and max values.
        x = data[0]
        yset = data[1]
        posx,posy = data[2]
        for vec in yset:
            self.miny = np.minimum(self.miny,vec.min())
            self.maxy = np.maximum(self.maxy,vec.max())
        # Get current plot coords.
        currCol = int(self.numPlots/self.numRows)
        currRow = np.mod(self.numPlots,self.numRows)
        self.pglayout.addPlot(row=currRow,
                              col=currCol,
                              title='Pos: ('+str(posx)+', '+str(posy)+')')
        # Plot data in specified plot.
        self.pglayout.getItem(currRow,currCol).plot(x,
                                                    yset.pop(0),
                                                    pen='w')
        for ind,y in enumerate(yset):
            cInd = np.mod(ind,self.numColors)
            self.pglayout.getItem(currRow,currCol).plot(x,
                                                        y,
                                                        clear=False,
                                                        pen=self.colors\
                                                            [:,cInd])
        self.numPlots+=1 # Increment number of plots.
        # Now update x and y ranges for all plots.
        for i in range(self.numPlots):
            c = int(i/self.numRows)
            r = np.mod(i,self.numRows)
            self.pglayout.getItem(r,c).setXRange(x.min(),
                                                 x.max(),
                                                 padding=0.05)
            self.pglayout.getItem(r,c).setYRange(self.miny,
                                                 self.maxy,
                                                 padding=0.05)
"""
Method to interpolate a datacube to be on the same grid as a reference 
datacube.
CALL ARGUMENTS:
    ref: 3-D ndarray. The reference datacube.
    cube: 3-D ndarray. The cube to be transformed. It will be put on the same
          grid in the X, Y, and Z directions as the ref cube.

RETURNS:
    tformed: 3-D ndarray. A cube with the same dimensions as ref.
"""
def interpCube(ref, cube):
    # We assume that the edges of the two images are the same. It's just the
    # spacing of the grid that is different.
    n,m,p = ref.shape
    q,r,s = cube.shape
    newx = np.linspace(0,cube.shape[0]-1,ref.shape[0])
    newy = np.linspace(0,cube.shape[1]-1,ref.shape[1])
    newz = np.linspace(0,cube.shape[2]-1,ref.shape[2])
    # Now get new coordinate system.
    inds = np.meshgrid(newx,newy,newz)
    coords = np.array([inds[0].ravel(),inds[1].ravel(),inds[2].ravel()])
    # Now interpolate onto this new coordinate system.
    tmp = map_coordinates(cube,coords,order=1,mode='nearest')
    # Reshape correctly.
    newx_count = np.arange(0,ref.shape[0])
    newy_count = np.arange(0,ref.shape[1])
    newz_count = np.arange(0,ref.shape[2])
    inds = np.meshgrid(newx_count,newy_count,newz_count)
    coords = np.array([inds[0].ravel(),inds[1].ravel(),inds[2].ravel()])
    tformed = np.empty_like(ref)
    tformed[coords[0,:],coords[1,:],coords[2,:]] = tmp
    
    return tformed
    

"""
Method to create an RGB representation of an input datacube.
CALL ARGUMENTS:
    cube: 3-D ndarray. The cube to be converted to RGB.
    region: 1-D ndarray (binary). A binary mask the same length as the 3rd
            dimension of cube. Wherever region is 1 will be used to make the
            RGB represetation.
    method: The way to do the conversion. Options are:
        'equal': Partition the datacube into 3 contiguous sections. If region
                 is not contiguous, the entries in cube where region is False
                 will be removed before the representation is computed. Sum all
                 entries within each section to get the R, G, and B components.
        'Si-excl': Uses the response of a CMV2K sensor to estimate what an
                   RGB sensor would see. If region is not contiguous, the
                   entries in cube where region is False will be removed before
                   the representation is computed. The first True value in
                   region will be treated as 400 nm, the last True value in
                   region will be treated as 700 nm. The True entries will be
                   assumed to be equally far apart from each other.
        'Si-incl': Same as Si-excl, but the entries in cube where region is
                   False are not removed. Instead, their Si response is set to
                   zero before the representation is computed.
    cubeX: 1-D ndarray. The x-values for the datacube's 3rd dimension. Only
           used for method='Si-excl' and method='Si-incl'. If not input, it is
           assumed that the range is 400-1000, equally spaced. If input, it
           should be the same length as the cube's 3rd dimension (after any
           effects from region).
RETURNS:
    rgb: The datacube as an rgb image.
NOTES:
    The CMV2K response curve for RGB sensors was retrieved from:
        "https://look.ams-osram.com/m/3cb99f54c7bbd621/original/
        CMV2000-Global-Shutter-CMOS-Image-Sensor-for-Machine-Vision.pdf"
        using "https://plotdigitizer.com/"
"""
def makeRGB(cube, region, method='equal', cubeX=None):
    if method == 'equal':
        # Select desired region of cube.
        cube = cube[:,:,region].astype(float)
        # Partition into 3 sections.
        n,m,p = cube.shape
        # Partition into sections and sum.
        # In the case of spectral data with units of nm, for RGB, we have to
        # put the blue part last and the red part first, so we do that here.
        rgb = np.zeros((n,m,3))
        rgb[:,:,2] = cube[:,:,:int(p/3)].mean(axis=2)
        rgb[:,:,1] = cube[:,:,int(p/3):2*int(p/3)].mean(axis=2)
        rgb[:,:,0] = cube[:,:,2*int(p/3):].mean(axis=2)
        
    elif method[:2] == 'Si':
        # Load in CMV2K spectral responses.
        redResponse = np.array(pd.read_csv('./CMV2K_response_red.csv'))
        greenResponse = np.array(pd.read_csv('./CMV2K_response_green1.csv'))
        blueResponse = np.array(pd.read_csv('./CMV2K_response_blue.csv'))
        # Make sure we have x-values.
        if cubeX is None:
            cubeX = np.linspace(400,1000,num=cube.shape[2])
        # If using Si-excl, then use region to select a section.
        if method == 'Si-excl':
            cube = cube[:,:,region].astype(float)
        # Put the response curves on the same grid as the input data.
        interpRed = interp1d(redResponse[:,0],
                             redResponse[:,1],
                             fill_value='extrapolate')
        redResponse = np.array([cubeX,
                                 interpRed(cubeX)]).T
        interpGreen = interp1d(greenResponse[:,0],
                               greenResponse[:,1],
                               fill_value='extrapolate')
        greenResponse = np.array([cubeX,
                                  interpGreen(cubeX)]).T
        interpBlue = interp1d(blueResponse[:,0],
                              blueResponse[:,1],
                              fill_value='extrapolate')
        blueResponse = np.array([cubeX,
                                 interpBlue(cubeX)]).T
        # If using Si-incl, zero-out the response where region is False
        if method == 'Si-incl':
            redResponse[~region,1] = 0
            greenResponse[~region,1] = 0
            blueResponse[~region,1] = 0
        # Now take the inner product of the response and the input, divided by
        # the length of xCube.
        n,m,p = cube.shape
        rgb = np.zeros((n,m,3))
        rgb[:,:,0] = (cube*redResponse[:,1]).mean(axis=2)
        rgb[:,:,1] = (cube*greenResponse[:,1]).mean(axis=2)
        rgb[:,:,2] = (cube*blueResponse[:,1]).mean(axis=2)
        
    return rgb
    
if __name__ == '__main__':
    """ Code Demo """
    # This hyperspectral datacube comes from the CAVE dataset.
    # https://cave.cs.columbia.edu/repository/Multispectral
    cubepath = "./jelly_beans_ms.npy" 
    cube = np.load(cubepath).astype(float)
    cubes = []
    for i in range(4):
        cubes.append(cube+i*1000)
    cubes = tuple(cubes)
    
    se = extractor()
    se(cubes,np.linspace(400,700,cubes[0].shape[2]))