import numpy
import glob
import pickle
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import random


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

	input=numpy.reshape(input_sequence_array,(len(input_sequence_array),200,1))
	input=input/float(notes_length)

	return(input_sequence_array,input)

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
	model.load_weights('weights-improvement-05-3.6357-bigger.hdf5')#use your latest weights here
	
	return model
	

def generateNewNotesFromModel(model,input_sequence_array,notes_length,notes_name):
	
	int_val_notes=dict((number,note) for number,note in enumerate(notes_name))
	
	sample_input=input_sequence_array[numpy.random.randint(0, len(input_sequence_array)-1)]
	
	generated_sequence=[]
	
	for i in range(300):
		generated_input= numpy.reshape(sample_input,(1, len(sample_input), 1))
		generated_input=generated_input/float(notes_length)
		next_sequence=model.predict(generated_input, verbose=0)
		temp=numpy.argmax(next_sequence)
		next_sequence_note=int_val_notes[temp]
		generated_sequence.append(next_sequence_note)
		
		sample_input.append(temp)
		sample_input=sample_input[1:len(sample_input)]
	
	return generated_sequence


def generateMidiAudio(generated_sequence):
	gap=0
	generated_notes=[]

	for sample_input in generated_sequence:
		if('.' in sample_input) or sample_input.isdigit():
			notes_found=sample_input.split('.')
			notes_array=[]
			for i in notes_found:
				next_note=note.Note(int(i))
				if(random.randint(1,3)==2):
					next_note.storedInstrument= instrument.Piano()
				else:
					next_note.storedInstrument= instrument.Guitar()
				notes_array.append(next_note)
			next_chord=chord.Chord(notes_array)
			next_chord.offset=gap
			generated_notes.append(next_chord)
		else:
			next_note=note.Note(sample_input)
			next_note.offset=gap
			next_note.storedInstrument=instrument.Piano()
			generated_notes.append(next_note)
		gap=gap+0.5
	
	generatedsequence=stream.Stream(generated_notes)
	generatedsequence.write('midi', fp='output_audio.mid')
	

def generateAudio():
	with open('InputNotes/notes','rb') as input_audio:
		notes_array=pickle.load(input_audio)
		
	notes_length=len(set(notes_array))
	notes_name = sorted(set(item for item in notes_array))
	input_sequence_array,input = generateNoteSequencesFromNotes(notes_array,notes_length)
	model=generateLSTM(input,notes_length)
	generated_sequence=generateNewNotesFromModel(model,input_sequence_array,notes_length,notes_name)
	generateMidiAudio(generated_sequence)

if __name__=='__main__':
		generateAudio()
	
	
	
	
	
	
	
	
	
	
	
	