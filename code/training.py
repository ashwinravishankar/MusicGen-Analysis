import numpy
import glob
import pickle
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def generateNotesFromMidi():
	notes_array=[]
	for input in glob.glob("InputFile/*.mid"):
		audio=converter.parse(input)
		print("Converting %s" % input)
		
		converted_notes=None
		try:
			partitioned_notes=instrument.partitionByInstrument(audio)
			converted_notes=partitioned_notes.parts[0].recurse()
		except:
			converted_notes=audio.flat.notes_array
		
		for portion in converted_notes:
			if isinstance(portion, chord.Chord):
				notes_array.append('.'.join(str(i) for i in portion.normalOrder))
			elif isinstance(portion, note.Note):
				notes_array.append(str(portion.pitch))
	with open('InputNotes/notes', 'wb') as input_notes:
		pickle.dump(notes_array, input_notes)
		
	return notes_array

def generateNoteSequencesFromNotes(notes_array,notes_length):
	
	note_names=sorted(set(item for item in notes_array))
	
	int_val_notes=dict((note,number) for number,note in enumerate(note_names))
	input_sequence_array=[]
	output_sequence_array=[]
	
	for i in range(0,len(notes_array)-200, 1):
		input_sequence=notes_array[i:i+200]
		output_sequence=notes_array[i+200]
		input_sequence_array.append([int_val_notes[char] for char in input_sequence])
		output_sequence_array.append(int_val_notes[output_sequence])
		
	input_sequence_array=numpy.reshape(input_sequence_array,(len(input_sequence_array),200,1))
	input_sequence_array=input_sequence_array/float(notes_length)
	
	output_sequence_array=np_utils.to_categorical(output_sequence_array)
	
	return(input_sequence_array,output_sequence_array)

	
def generateLSTM(input_sequence_array,notes_length):
	model = Sequential()
	model.add(LSTM(64,input_shape=(input_sequence_array.shape[1], input_sequence_array.shape[2]),return_sequences=True))
	model.add(Dropout(0.5))
	model.add(LSTM(64, return_sequences=True))
	model.add(Dropout(0.5))
	model.add(LSTM(64))
	model.add(Dense(256))
	model.add(Dropout(0.5))
	model.add(Dense(notes_length))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	return model



def modelTraining():
	notes_array= generateNotesFromMidi()
	
	notes_length=len(set(notes_array))
	input_sequence_array,output_sequence_array= generateNoteSequencesFromNotes(notes_array,notes_length)
	model=generateLSTM(input_sequence_array,notes_length)
	
	networkTraining(model,input_sequence_array,output_sequence_array)


def networkTraining(model,input_sequence_array,output_sequence_array):
	filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
	checkpoint = ModelCheckpoint(
		filepath,
		monitor='loss',
		verbose=0,
		save_best_only=True,
		mode='min'
	)
	callbacks_list = [checkpoint]

	model.fit(input_sequence_array,output_sequence_array, epochs=50, batch_size=64, callbacks=callbacks_list)


if __name__=='__main__':
	modelTraining()
