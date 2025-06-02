# Datacube-Extractor
Code for a GUI to visualize and save the vectors attached to X-Y points in a datacube. This was designed with spectral or polarimetric imaging in mind, but can be used to visualize any datacube.

To run the code, put the datacubes to extract from into a tuple, define a np array to be the x-axis for the first datacube in the tuple, and specify the region you want to use for visualization as a boolean array the same length as the 3rd dimension of the first cube in the tuple (where True means "use this entry" and False means "do not use this entry"). All datacubes in the tuple will be interpolated to be the same dimensions as the first one.

Next, instantiate the extractor object. It has two kwargs: numRows and rgbType. numRows determines the number of rows to display in the "Saved Vectors" display before creating a new column. rgbType determines how to convert the datacube to RGB for display and can be "equal" or "Si".  For any non-spectral use case, use "equal". For spectral imaging in the VNIR, "Si" can be used to simulate how the image would appear on a CMV2000 focal plane with RGB filters on it (see below plot "Plot_CMV2K_response.png" the used absorption spectra).
![Plot_CMV2K_response](https://github.com/user-attachments/assets/05cbd283-177a-498c-894d-b7afead6ad47)

To run the extractor, call it as demonstrated here.
![Screenshot 2025-06-02 113346](https://github.com/user-attachments/assets/79c21bd1-82ed-4a5b-ac12-cfd7aab38058)

As you drag your cursor over the image, the vector at your cursor location will be displayed in the "Vector Display" window. Left-clicking on any point in the image will save the vector to an archive, which is returned when all windows are closed. The vector will then be added to the "Saved Vectors" window. Right-clicking anywhere in the image will remove the most recent vector from the archive and remove its plot from the "Saved Vectors" window.
![Screenshot 2025-06-02 112952](https://github.com/user-attachments/assets/4d03f227-f0c3-43a2-8ccd-c17f4a6d8746)
