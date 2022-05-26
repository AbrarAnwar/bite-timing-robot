#!/usr/bin/env python

from gtts import gTTS
import os
import playsound
import rospy
from std_msgs.msg import String

def speak(msg):
    # create tts object
    text = msg.data
    print("Recieved text: " + text)
    tts = gTTS(text=text)

    # if temp.mp3 exists, delete
    if os.path.isfile('temp.mp3'):
        os.remove('temp.mp3')

    filename = "temp.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

def main():
    # create ros node
    rospy.init_node('play_sound_node')

    # create subscriber
    rospy.Subscriber('/robot/speech', String, speak)

    # spin
    rospy.spin()

if __name__ == '__main__':
    main()