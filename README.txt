Submission:
	1. README
	2. Model.h5
	3. drive.py
	4. submission.IPYNB
	5. output_video.mp4
	6. autodrive.py


Discription:
	1. submission.ipynb:
		File describes my approach and thoughts behind processing training data. It creates the array for trainig the model
	2. autodrive.py:
		File contains the model. It reads the numpy arrays produced by the submission.ipynb. in the end, it prints the summary of the model, along with saving the 		model.
	3. model.h5:
		keras saved model.
	4. ran_smooth.mp4:
		Video of the output
	5. drive.py:
		for running the simulation with model.h5. This file is tweaked to convert to 'YUV' space and reduce the image resolution to match the image size with the 		NVIDIA paper [1].

