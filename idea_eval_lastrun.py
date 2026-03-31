#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.0),
    on March 31, 2026, at 11:48
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.0'
expName = 'idea_eval'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participantID': ["A800", "A111", "B870"],
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participantID'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\aivan\\Documents\\source_monitor\\idea_eval_lastrun.py',
        savePickle=False, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=True,
            monitor='fortis_m', color=(0.8824, 1.0000, 1.0000), colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = (0.8824, 1.0000, 1.0000)
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # enter 'rush' mode (raise CPU priority)
    if not PILOTING:
        core.rush(enable=True)
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "startup_routine" ---
    # Run 'Begin Experiment' code from startupCode
    import csv
    
    participantId = expInfo['participantID']
    stimuliPath = f'./stimuli/{participantId}_ideas.csv'
    
    with open(stimuliPath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        allIdeas = list(reader)
    
    totalIdeas = len(allIdeas)
    
    ideaStims = []
    for i in range(10):
        stim = visual.TextStim(
            win,
            text='',
            pos=(0, 0.15 - i * 0.08),
            height=0.04,
            font='Courier New',
            color='#090521',
            anchorHoriz='left',
            alignText='left',
            wrapWidth=0.75
        )
        ideaStims.append(stim)
    
    ideaStimsObs = []
    for i in range(10):
        stim = visual.TextStim(
            win,
            text='',
            pos=(0.1, 0.15 - i * 0.08),
            height=0.04,
            font='Courier New',
            color='#090521',
            anchorHoriz='left',
            alignText='left',
            wrapWidth=0.75
        )
        ideaStimsObs.append(stim)
    
    headerLeft = visual.TextStim(
        win, text='Участник А', font='Courier New',
        pos=(-0.50, 0.27), height=0.045,
        color='#090521', bold=True
    )
    headerRight = visual.TextStim(
        win, text='Участник Б', font='Courier New',
        pos=(0.50, 0.27), height=0.045,
        color='#090521', bold=True
    )
    
    nextButton = visual.Rect(
        win, width=0.3, height=0.07,
        pos=(0, -0.40), fillColor='#003049'
    )
    nextButtonText = visual.TextStim(
        win, text='Далее', font='Courier New',
        pos=(0, -0.40), height=0.05, color='#f1faee'
    )
    
    selectionMouse = event.Mouse(win=win)
    
    print(f'[startupCode] Загружено листов: {totalIdeas} для участника {participantId}')
    # Run 'Begin Experiment' code from initialiseConditions
    skipConfidence = False
    skipSource = False
    ceInstructionCount = 0
    isSourceEmpty = False
    
    prevBaseItem = ''
    currentBaseItem = ''
    prevRole = ''
    currentRole = ''
    
    selectedObs = []
    wasPressedObs = False
    
    # --- Initialize components for Routine "instructions" ---
    inastructionText = visual.TextStim(win=win, name='inastructionText',
        text='Добро пожаловать на второй этап исследования!\n\nСегодня вам предстоит поработать с материалами  предыдущей встречи. \n\nВы увидите списки идей, которые предлагались участниками вашей группы.\nВаша задача — оценить эти идеи по нескольким критериям. \n\nНажмите [ПРОБЕЛ], чтобы продолжить.',
        font='Courier New',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=1.3, ori=0.0, 
        color=(-0.9294, -0.9608, -0.7412), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instructionsButton = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "instructions_part_1" ---
    insructionFirstPartText = visual.TextStim(win=win, name='insructionFirstPartText',
        text='В этой части вам будут показаны списки идей, записанных участниками вашей группы.\n\nВаша задача — отметить те идеи, которые, по вашим воспоминаниям, действительно звучали на предыдущей встрече. \n\nКликните на идею, чтобы отметить её. Отмеченная идея выделится зелёным цветом. Повторный клик снимает отметку.\n\nКогда закончите, нажмите кнопку «Далее».\n\nНажмите [ПРОБЕЛ] для начала.',
        font='Courier New',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=1.3, ori=0.0, 
        color=(-0.9294, -0.9608, -0.7412), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instructionsFirstPartButton = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "routine_instructions" ---
    routineInstructionsText = visual.TextStim(win=win, name='routineInstructionsText',
        text='На предыдущей встрече вы предлагали идеи для этого предмета вместе с партнёром:',
        font='Courier New',
        pos=(0, 0.20), draggable=False, height=0.04, wrapWidth=1.3, ori=0.0, 
        color=(-0.9294, -0.9608, -0.7412), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    coreItemText = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Courier New',
         ori=0.0, pos=(0, 0.07), draggable=False,      letterHeight=0.06,
         size=(0.5, 0.5), borderWidth=0.0,
         color=(-0.9294, -0.9608, -0.7412), colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='coreItemText',
         depth=-2, autoLog=False,
    )
    participantRoleText = visual.TextStim(win=win, name='participantRoleText',
        text='Вы выступали в роли:',
        font='Courier New',
        pos=(0, -0.10), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color=(-0.9294, -0.9608, -0.7412), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    roleText = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Courier New',
         ori=0.0, pos=(0, -0.17), draggable=False,      letterHeight=0.06,
         size=(0.5, 0.5), borderWidth=0.0,
         color=(-0.9294, -0.9608, -0.7412), colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='roleText',
         depth=-4, autoLog=False,
    )
    routineInstructionsButtonText = visual.TextStim(win=win, name='routineInstructionsButtonText',
        text='Нажмите [ПРОБЕЛ] для начала.',
        font='Courier New',
        pos=(0, -0.30), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=(-0.9294, -0.9608, -0.7412), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    coreItemButton = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "item_display" ---
    ideaQuestionText = visual.TextStim(win=win, name='ideaQuestionText',
        text='Отметьте именно те идеи, которые, по Вашему мнению, действительно звучали на встрече неделю назад',
        font='Courier New',
        pos=(0, 0.40), draggable=False, height=0.05, wrapWidth=1.3, ori=0.0, 
        color=(-0.9294, -0.9608, -0.7412), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    showIdeaProceedButton = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "confidence_evaluation" ---
    confidenceText = visual.TextStim(win=win, name='confidenceText',
        text='Насколько вы уверены в том, что именно Ваш партнер является автором данной идеи?',
        font='Courier New',
        pos=(0, 0.30), draggable=False, height=0.04, wrapWidth=1.3, ori=0.0, 
        color=(-0.9294, -0.9608, -0.7412), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from changeObserverText
    if currentRole != 'observation':
        print('Current role: OBSERVER')
        confidenceText.text = 'Насколько вы уверены в том, что именно этот участник является автором данной идеи?'
    confidenceIdeaText = visual.TextBox2(
         win, text='...', placeholder='Type here...', font='Courier New',
         ori=0.0, pos=(0, 0.10), draggable=False,      letterHeight=0.05,
         size=(1.3, 0.5), borderWidth=0.0,
         color=(-0.9294, -0.9608, -0.7412), colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='confidenceIdeaText',
         depth=-2, autoLog=False,
    )
    confidenceTextHint = visual.TextStim(win=win, name='confidenceTextHint',
        text='0% — Идея, навеянная другой идеей      100% — Точно оригинальная идея',
        font='Courier New',
        pos=(0, -0.07), draggable=False, height=0.03, wrapWidth=1.8, ori=0.0, 
        color=(-0.9294, -0.9608, -0.7412), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    confidenceSlider = visual.Slider(win=win, name='confidenceSlider',
        startValue=None, size=(1.0, 0.06), pos=(0, -0.25), units=win.units,
        labels=('0%', '50%', '100%'), ticks=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), granularity=0.0,
        style='rating', styleTweaks=[], opacity=None,
        labelColor='LightGray', markerColor=(-0.4353, -0.5216, 0.0902), lineColor=(0.6000, 0.6941, 0.8510), colorSpace='rgb',
        font='Courier New', labelHeight=0.05,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    buttonText = visual.TextStim(win=win, name='buttonText',
        text='Нажмите ENTER для подтверждения',
        font='Courier New',
        pos=(0, -0.4), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color=(-0.9294, -0.9608, -0.7412), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    confidenceProceedButton = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "prepare_source_routine" ---
    
    # --- Initialize components for Routine "source_evaluation" ---
    sourceIdeaLabel = visual.TextBox2(
         win, text='...', placeholder='Type here...', font='Courier New',
         ori=0.0, pos=(0, 0.30), draggable=False,      letterHeight=0.05,
         size=(1.3, 0.5), borderWidth=0.0,
         color=(-0.9294, -0.9608, -0.7412), colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='sourceIdeaLabel',
         depth=0, autoLog=False,
    )
    sourceEvaluationText = visual.TextStim(win=win, name='sourceEvaluationText',
        text='Вы отметили, что эта идея, по вашему мнению,  была навеяна другой идеей из обсуждения.\n \nОпишите, какая идея, на ваш взгляд, могла послужить для неё отправной точкой:',
        font='Courier New',
        pos=(0, 0.15), draggable=False, height=0.04, wrapWidth=1.3, ori=0.0, 
        color=(-0.9294, -0.9608, -0.7412), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    sourceEvaluationTextBox = visual.TextBox2(
         win, text=None, placeholder='Введите ваш ответ здесь...', font='Courier New',
         ori=0.0, pos=(0, -0.15), draggable=False,      letterHeight=0.05,
         size=(1.4, 0.2), borderWidth=1.0,
         color=(-0.9294, -0.9608, -0.7412), colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=(0.5451, 0.7647, 0.8745), borderColor=(0.4353, 0.6314, 0.7804),
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='sourceEvaluationTextBox',
         depth=-2, autoLog=False,
    )
    sourceButtonText = visual.TextStim(win=win, name='sourceButtonText',
        text='Нажмите ENTER для подтверждения',
        font='Courier New',
        pos=(0, -0.4), draggable=False, height=0.035, wrapWidth=1.3, ori=0.0, 
        color=(-0.9294, -0.9608, -0.7412), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    sourceEvaluationButton = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "thank_you" ---
    thankYouText = visual.TextStim(win=win, name='thankYouText',
        text='Благодарим за участие!',
        font='Courier New',
        pos=(0, 0), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "startup_routine" ---
    # create an object to store info about Routine startup_routine
    startup_routine = data.Routine(
        name='startup_routine',
        components=[],
    )
    startup_routine.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for startup_routine
    startup_routine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    startup_routine.tStart = globalClock.getTime(format='float')
    startup_routine.status = STARTED
    startup_routine.maxDuration = None
    # keep track of which components have finished
    startup_routineComponents = startup_routine.components
    for thisComponent in startup_routine.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "startup_routine" ---
    thisExp.currentRoutine = startup_routine
    startup_routine.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=startup_routine,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            startup_routine.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if startup_routine.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in startup_routine.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "startup_routine" ---
    for thisComponent in startup_routine.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for startup_routine
    startup_routine.tStop = globalClock.getTime(format='float')
    startup_routine.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "startup_routine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions" ---
    # create an object to store info about Routine instructions
    instructions = data.Routine(
        name='instructions',
        components=[inastructionText, instructionsButton],
    )
    instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instructionsButton
    instructionsButton.keys = []
    instructionsButton.rt = []
    _instructionsButton_allKeys = []
    # store start times for instructions
    instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions.tStart = globalClock.getTime(format='float')
    instructions.status = STARTED
    instructions.maxDuration = None
    # keep track of which components have finished
    instructionsComponents = instructions.components
    for thisComponent in instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    thisExp.currentRoutine = instructions
    instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *inastructionText* updates
        
        # if inastructionText is starting this frame...
        if inastructionText.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            inastructionText.frameNStart = frameN  # exact frame index
            inastructionText.tStart = t  # local t and not account for scr refresh
            inastructionText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(inastructionText, 'tStartRefresh')  # time at next scr refresh
            # update status
            inastructionText.status = STARTED
            inastructionText.setAutoDraw(True)
        
        # if inastructionText is active this frame...
        if inastructionText.status == STARTED:
            # update params
            pass
        
        # *instructionsButton* updates
        
        # if instructionsButton is starting this frame...
        if instructionsButton.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructionsButton.frameNStart = frameN  # exact frame index
            instructionsButton.tStart = t  # local t and not account for scr refresh
            instructionsButton.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructionsButton, 'tStartRefresh')  # time at next scr refresh
            # update status
            instructionsButton.status = STARTED
            # keyboard checking is just starting
            instructionsButton.clock.reset()  # now t=0
        if instructionsButton.status == STARTED:
            theseKeys = instructionsButton.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instructionsButton_allKeys.extend(theseKeys)
            if len(_instructionsButton_allKeys):
                instructionsButton.keys = _instructionsButton_allKeys[-1].name  # just the last key pressed
                instructionsButton.rt = _instructionsButton_allKeys[-1].rt
                instructionsButton.duration = _instructionsButton_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions
    instructions.tStop = globalClock.getTime(format='float')
    instructions.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_part_1" ---
    # create an object to store info about Routine instructions_part_1
    instructions_part_1 = data.Routine(
        name='instructions_part_1',
        components=[insructionFirstPartText, instructionsFirstPartButton],
    )
    instructions_part_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instructionsFirstPartButton
    instructionsFirstPartButton.keys = []
    instructionsFirstPartButton.rt = []
    _instructionsFirstPartButton_allKeys = []
    # store start times for instructions_part_1
    instructions_part_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_part_1.tStart = globalClock.getTime(format='float')
    instructions_part_1.status = STARTED
    instructions_part_1.maxDuration = None
    # keep track of which components have finished
    instructions_part_1Components = instructions_part_1.components
    for thisComponent in instructions_part_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_part_1" ---
    thisExp.currentRoutine = instructions_part_1
    instructions_part_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *insructionFirstPartText* updates
        
        # if insructionFirstPartText is starting this frame...
        if insructionFirstPartText.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            insructionFirstPartText.frameNStart = frameN  # exact frame index
            insructionFirstPartText.tStart = t  # local t and not account for scr refresh
            insructionFirstPartText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(insructionFirstPartText, 'tStartRefresh')  # time at next scr refresh
            # update status
            insructionFirstPartText.status = STARTED
            insructionFirstPartText.setAutoDraw(True)
        
        # if insructionFirstPartText is active this frame...
        if insructionFirstPartText.status == STARTED:
            # update params
            pass
        
        # *instructionsFirstPartButton* updates
        
        # if instructionsFirstPartButton is starting this frame...
        if instructionsFirstPartButton.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructionsFirstPartButton.frameNStart = frameN  # exact frame index
            instructionsFirstPartButton.tStart = t  # local t and not account for scr refresh
            instructionsFirstPartButton.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructionsFirstPartButton, 'tStartRefresh')  # time at next scr refresh
            # update status
            instructionsFirstPartButton.status = STARTED
            # keyboard checking is just starting
            instructionsFirstPartButton.clock.reset()  # now t=0
        if instructionsFirstPartButton.status == STARTED:
            theseKeys = instructionsFirstPartButton.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instructionsFirstPartButton_allKeys.extend(theseKeys)
            if len(_instructionsFirstPartButton_allKeys):
                instructionsFirstPartButton.keys = _instructionsFirstPartButton_allKeys[-1].name  # just the last key pressed
                instructionsFirstPartButton.rt = _instructionsFirstPartButton_allKeys[-1].rt
                instructionsFirstPartButton.duration = _instructionsFirstPartButton_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructions_part_1,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructions_part_1.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructions_part_1.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructions_part_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_part_1" ---
    for thisComponent in instructions_part_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_part_1
    instructions_part_1.tStop = globalClock.getTime(format='float')
    instructions_part_1.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "instructions_part_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=totalIdeas, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "routine_instructions" ---
        # create an object to store info about Routine routine_instructions
        routine_instructions = data.Routine(
            name='routine_instructions',
            components=[routineInstructionsText, coreItemText, participantRoleText, roleText, routineInstructionsButtonText, coreItemButton],
        )
        routine_instructions.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from getIdeaData
        currentIndex = currentLoop.thisN
        ideaData = allIdeas[currentIndex]
        currentRole = ideaData['blockType']
        
        currentIdeas = []
        for i in range(1, 21):
            key = f'idea_{i}'
            val = ideaData.get(key, '').strip()
            if val:
                currentIdeas.append(val)
        
        currentIdeasObs = []
        if currentRole == 'observation':
            for i in range(1, 21):
                val = ideaData.get(f'idea_b_{i}', '').strip()
                if val:
                    currentIdeasObs.append(val)
        
        rememberedIdeas = []
        confidenceScores = {}
        sourceTexts = {}
        coreItemText.reset()
        coreItemText.setText(currentBaseItem)
        roleText.reset()
        roleText.setText(currentRole)
        # create starting attributes for coreItemButton
        coreItemButton.keys = []
        coreItemButton.rt = []
        _coreItemButton_allKeys = []
        # Run 'Begin Routine' code from onBaseItemChange
        currentIndex = currentLoop.thisN
        ideaData = allIdeas[currentIndex]
        
        currentBaseItem = ideaData['baseItem']
        currentRole = ideaData['blockType']
        
        if currentBaseItem == prevBaseItem and currentRole == prevRole:
            continueRoutine = False
        else:
            coreItemText.text = f'{currentBaseItem}'
            roleText.text = 'Наблюдатель' if currentRole == 'observation' else 'Генератор идей'
        # store start times for routine_instructions
        routine_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        routine_instructions.tStart = globalClock.getTime(format='float')
        routine_instructions.status = STARTED
        routine_instructions.maxDuration = None
        # keep track of which components have finished
        routine_instructionsComponents = routine_instructions.components
        for thisComponent in routine_instructions.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "routine_instructions" ---
        thisExp.currentRoutine = routine_instructions
        routine_instructions.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *routineInstructionsText* updates
            
            # if routineInstructionsText is starting this frame...
            if routineInstructionsText.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                routineInstructionsText.frameNStart = frameN  # exact frame index
                routineInstructionsText.tStart = t  # local t and not account for scr refresh
                routineInstructionsText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(routineInstructionsText, 'tStartRefresh')  # time at next scr refresh
                # update status
                routineInstructionsText.status = STARTED
                routineInstructionsText.setAutoDraw(True)
            
            # if routineInstructionsText is active this frame...
            if routineInstructionsText.status == STARTED:
                # update params
                pass
            
            # *coreItemText* updates
            
            # if coreItemText is starting this frame...
            if coreItemText.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                coreItemText.frameNStart = frameN  # exact frame index
                coreItemText.tStart = t  # local t and not account for scr refresh
                coreItemText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(coreItemText, 'tStartRefresh')  # time at next scr refresh
                # update status
                coreItemText.status = STARTED
                coreItemText.setAutoDraw(True)
            
            # if coreItemText is active this frame...
            if coreItemText.status == STARTED:
                # update params
                pass
            
            # *participantRoleText* updates
            
            # if participantRoleText is starting this frame...
            if participantRoleText.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                participantRoleText.frameNStart = frameN  # exact frame index
                participantRoleText.tStart = t  # local t and not account for scr refresh
                participantRoleText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(participantRoleText, 'tStartRefresh')  # time at next scr refresh
                # update status
                participantRoleText.status = STARTED
                participantRoleText.setAutoDraw(True)
            
            # if participantRoleText is active this frame...
            if participantRoleText.status == STARTED:
                # update params
                pass
            
            # *roleText* updates
            
            # if roleText is starting this frame...
            if roleText.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                roleText.frameNStart = frameN  # exact frame index
                roleText.tStart = t  # local t and not account for scr refresh
                roleText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(roleText, 'tStartRefresh')  # time at next scr refresh
                # update status
                roleText.status = STARTED
                roleText.setAutoDraw(True)
            
            # if roleText is active this frame...
            if roleText.status == STARTED:
                # update params
                pass
            
            # *routineInstructionsButtonText* updates
            
            # if routineInstructionsButtonText is starting this frame...
            if routineInstructionsButtonText.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                routineInstructionsButtonText.frameNStart = frameN  # exact frame index
                routineInstructionsButtonText.tStart = t  # local t and not account for scr refresh
                routineInstructionsButtonText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(routineInstructionsButtonText, 'tStartRefresh')  # time at next scr refresh
                # update status
                routineInstructionsButtonText.status = STARTED
                routineInstructionsButtonText.setAutoDraw(True)
            
            # if routineInstructionsButtonText is active this frame...
            if routineInstructionsButtonText.status == STARTED:
                # update params
                pass
            
            # *coreItemButton* updates
            
            # if coreItemButton is starting this frame...
            if coreItemButton.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                coreItemButton.frameNStart = frameN  # exact frame index
                coreItemButton.tStart = t  # local t and not account for scr refresh
                coreItemButton.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(coreItemButton, 'tStartRefresh')  # time at next scr refresh
                # update status
                coreItemButton.status = STARTED
                # keyboard checking is just starting
                coreItemButton.clock.reset()  # now t=0
            if coreItemButton.status == STARTED:
                theseKeys = coreItemButton.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _coreItemButton_allKeys.extend(theseKeys)
                if len(_coreItemButton_allKeys):
                    coreItemButton.keys = _coreItemButton_allKeys[-1].name  # just the last key pressed
                    coreItemButton.rt = _coreItemButton_allKeys[-1].rt
                    coreItemButton.duration = _coreItemButton_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=routine_instructions,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                routine_instructions.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if routine_instructions.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in routine_instructions.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "routine_instructions" ---
        for thisComponent in routine_instructions.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for routine_instructions
        routine_instructions.tStop = globalClock.getTime(format='float')
        routine_instructions.tStopRefresh = tThisFlipGlobal
        # Run 'End Routine' code from onBaseItemChange
        prevBaseItem = currentBaseItem
        prevRole = currentRole
        # the Routine "routine_instructions" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "item_display" ---
        # create an object to store info about Routine item_display
        item_display = data.Routine(
            name='item_display',
            components=[ideaQuestionText, showIdeaProceedButton],
        )
        item_display.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from getCurrentIdeas
        currentIndex = currentLoop.thisN
        ideaData = allIdeas[currentIndex]
        
        print('[getCurrentIdeas] Загружены следующие идеи для участника:', currentIdeas, currentIdeasObs if len(currentIdeasObs) > 0 else '')
        
        isObserver = (currentRole == 'observation')
        selected = [False] * len(currentIdeas)
        selectedObs = [False] * len(currentIdeasObs)
        wasPressed = False
        
        if isObserver:
            for i, stim in enumerate(ideaStims):
                stim.pos = (-0.7, 0.15 - i * 0.08)
                stim.wrapWidth = 0.75
                if i < len(currentIdeas):
                    stim.text = f'{i+1}. {currentIdeas[i]}'
                    stim.opacity = 1.0
                else:
                    stim.text = ''
                    stim.opacity = 0.0
        
            for i, stim in enumerate(ideaStimsObs):
                if i < len(currentIdeasObs):
                    stim.text = f'{i+1}. {currentIdeasObs[i]}'
                    stim.opacity = 1.0
                else:
                    stim.text = ''
                    stim.opacity = 0.0
        
        else:
            for i, stim in enumerate(ideaStims):
                stim.pos = (-0.35, 0.15 - i * 0.08)
                stim.wrapWidth = 1.1
                if i < len(currentIdeas):
                    stim.text = f'{i+1}. {currentIdeas[i]}'
                    stim.opacity = 1.0
                else:
                    stim.text = ''
                    stim.opacity = 0.0
        
            for stim in ideaStimsObs:
                stim.text = ''
                stim.opacity = 0.0
        
        
        selectionMouse.clickReset()
        
        currentSource = ideaData['sourceParticipant']
        currentBlockType = ideaData['blockType']
        # create starting attributes for showIdeaProceedButton
        showIdeaProceedButton.keys = []
        showIdeaProceedButton.rt = []
        _showIdeaProceedButton_allKeys = []
        # store start times for item_display
        item_display.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        item_display.tStart = globalClock.getTime(format='float')
        item_display.status = STARTED
        item_display.maxDuration = None
        # keep track of which components have finished
        item_displayComponents = item_display.components
        for thisComponent in item_display.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "item_display" ---
        thisExp.currentRoutine = item_display
        item_display.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from getCurrentIdeas
            buttons, times = selectionMouse.getPressed(getTime=True)
            
            if buttons[0]:
                if not wasPressed:
                    wasPressed = True
                    mousePos = selectionMouse.getPos()
            
                    for i, stim in enumerate(ideaStims):
                        if i < len(currentIdeas) and stim.opacity > 0:
                            if stim.boundingBox is not None:
                                if stim.contains(mousePos):
                                    selected[i] = not selected[i]
                    if isObserver:
                        for i, stim in enumerate(ideaStimsObs):
                            if i < len(currentIdeasObs) and stim.opacity > 0:
                                if stim.contains(mousePos):
                                    selectedObs[i] = not selectedObs[i]
            
                    if nextButton.contains(mousePos):
                        continueRoutine = False
            
            else:
                wasPressed = False
            
            for i, stim in enumerate(ideaStims):
                if i < len(currentIdeas):
                    stim.color = '#00c853' if selected[i] else '#090521'
                stim.draw()
            
            if isObserver:
                headerLeft.draw()
                headerRight.draw()
                for i, stim in enumerate(ideaStimsObs):
                    if stim.opacity > 0:
                        stim.color = '#00c853' if (i < len(selectedObs) and selectedObs[i]) else '#090521'
                    stim.draw()
            
            nextButton.draw()
            nextButtonText.draw()
            
            # *ideaQuestionText* updates
            
            # if ideaQuestionText is starting this frame...
            if ideaQuestionText.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ideaQuestionText.frameNStart = frameN  # exact frame index
                ideaQuestionText.tStart = t  # local t and not account for scr refresh
                ideaQuestionText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ideaQuestionText, 'tStartRefresh')  # time at next scr refresh
                # update status
                ideaQuestionText.status = STARTED
                ideaQuestionText.setAutoDraw(True)
            
            # if ideaQuestionText is active this frame...
            if ideaQuestionText.status == STARTED:
                # update params
                pass
            
            # *showIdeaProceedButton* updates
            
            # if showIdeaProceedButton is starting this frame...
            if showIdeaProceedButton.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                showIdeaProceedButton.frameNStart = frameN  # exact frame index
                showIdeaProceedButton.tStart = t  # local t and not account for scr refresh
                showIdeaProceedButton.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(showIdeaProceedButton, 'tStartRefresh')  # time at next scr refresh
                # update status
                showIdeaProceedButton.status = STARTED
                # keyboard checking is just starting
                showIdeaProceedButton.clock.reset()  # now t=0
            if showIdeaProceedButton.status == STARTED:
                theseKeys = showIdeaProceedButton.getKeys(keyList=['y', 'n'], ignoreKeys=["escape"], waitRelease=False)
                _showIdeaProceedButton_allKeys.extend(theseKeys)
                if len(_showIdeaProceedButton_allKeys):
                    showIdeaProceedButton.keys = _showIdeaProceedButton_allKeys[-1].name  # just the last key pressed
                    showIdeaProceedButton.rt = _showIdeaProceedButton_allKeys[-1].rt
                    showIdeaProceedButton.duration = _showIdeaProceedButton_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=item_display,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                item_display.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if item_display.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in item_display.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "item_display" ---
        for thisComponent in item_display.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for item_display
        item_display.tStop = globalClock.getTime(format='float')
        item_display.tStopRefresh = tThisFlipGlobal
        # Run 'End Routine' code from getCurrentIdeas
        rememberedIdeas = [
            currentIdeas[i]
            for i in range(len(currentIdeas))
            if selected[i]
        ]
        
        if isObserver:
            rememberedIdeas += [
                currentIdeasObs[i]
                for i in range(len(currentIdeasObs))
                if selectedObs[i]
            ]
        
        for i, idea in enumerate(currentIdeas):
            thisExp.addData(f'list_{trials.thisN+1}_idea_{i+1}', idea)
            thisExp.addData(
                f'list_{trials.thisN+1}_remembered_{i+1}',
                'yes' if selected[i] else 'no'
            )
        
        if isObserver:
            for i, idea in enumerate(currentIdeasObs):
                thisExp.addData(f'list_{trials.thisN+1}_idea_b_{i+1}', idea)
                thisExp.addData(
                    f'list_{trials.thisN+1}_remembered_b_{i+1}',
                    'yes' if selectedObs[i] else 'no'
                )
        
        confidenceIndex = 0
        
        print('[pushToArr] Участник отметил следующие идеи:',rememberedIdeas)
        # Run 'End Routine' code from storeResponseKey
        response = showIdeaProceedButton.keys
        if response == 'n':
            skipConfidence = True
            skipSource = True
        else:
            skipConfidence = False
            skipSource = False
        
        thisExp.addData('sourceParticipant', currentSource)
        thisExp.addData('blockType', currentBlockType)
        thisExp.addData('remembered', response)
        # the Routine "item_display" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        confidence_loop = data.TrialHandler2(
            name='confidence_loop',
            nReps=len(rememberedIdeas), 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(confidence_loop)  # add the loop to the experiment
        thisConfidence_loop = confidence_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisConfidence_loop.rgb)
        if thisConfidence_loop != None:
            for paramName in thisConfidence_loop:
                globals()[paramName] = thisConfidence_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisConfidence_loop in confidence_loop:
            confidence_loop.status = STARTED
            if hasattr(thisConfidence_loop, 'status'):
                thisConfidence_loop.status = STARTED
            currentLoop = confidence_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisConfidence_loop.rgb)
            if thisConfidence_loop != None:
                for paramName in thisConfidence_loop:
                    globals()[paramName] = thisConfidence_loop[paramName]
            
            # --- Prepare to start Routine "confidence_evaluation" ---
            # create an object to store info about Routine confidence_evaluation
            confidence_evaluation = data.Routine(
                name='confidence_evaluation',
                components=[confidenceText, confidenceIdeaText, confidenceTextHint, confidenceSlider, buttonText, confidenceProceedButton],
            )
            confidence_evaluation.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            confidenceIdeaText.reset()
            # Run 'Begin Routine' code from skipConfidenceRoutine
            if len(rememberedIdeas) == 0:
                continueRoutine = False
            else:
                currentСonfIdea = rememberedIdeas[confidence_loop.thisN]
                confidenceIdeaText.text = (f'«{currentСonfIdea}»')
                confidenceSlider.reset()
            confidenceSlider.reset()
            # create starting attributes for confidenceProceedButton
            confidenceProceedButton.keys = []
            confidenceProceedButton.rt = []
            _confidenceProceedButton_allKeys = []
            # store start times for confidence_evaluation
            confidence_evaluation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            confidence_evaluation.tStart = globalClock.getTime(format='float')
            confidence_evaluation.status = STARTED
            confidence_evaluation.maxDuration = None
            # keep track of which components have finished
            confidence_evaluationComponents = confidence_evaluation.components
            for thisComponent in confidence_evaluation.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "confidence_evaluation" ---
            thisExp.currentRoutine = confidence_evaluation
            confidence_evaluation.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisConfidence_loop, 'status') and thisConfidence_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *confidenceText* updates
                
                # if confidenceText is starting this frame...
                if confidenceText.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    confidenceText.frameNStart = frameN  # exact frame index
                    confidenceText.tStart = t  # local t and not account for scr refresh
                    confidenceText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(confidenceText, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    confidenceText.status = STARTED
                    confidenceText.setAutoDraw(True)
                
                # if confidenceText is active this frame...
                if confidenceText.status == STARTED:
                    # update params
                    pass
                
                # *confidenceIdeaText* updates
                
                # if confidenceIdeaText is starting this frame...
                if confidenceIdeaText.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    confidenceIdeaText.frameNStart = frameN  # exact frame index
                    confidenceIdeaText.tStart = t  # local t and not account for scr refresh
                    confidenceIdeaText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(confidenceIdeaText, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    confidenceIdeaText.status = STARTED
                    confidenceIdeaText.setAutoDraw(True)
                
                # if confidenceIdeaText is active this frame...
                if confidenceIdeaText.status == STARTED:
                    # update params
                    pass
                
                # *confidenceTextHint* updates
                
                # if confidenceTextHint is starting this frame...
                if confidenceTextHint.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    confidenceTextHint.frameNStart = frameN  # exact frame index
                    confidenceTextHint.tStart = t  # local t and not account for scr refresh
                    confidenceTextHint.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(confidenceTextHint, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    confidenceTextHint.status = STARTED
                    confidenceTextHint.setAutoDraw(True)
                
                # if confidenceTextHint is active this frame...
                if confidenceTextHint.status == STARTED:
                    # update params
                    pass
                
                # *confidenceSlider* updates
                
                # if confidenceSlider is starting this frame...
                if confidenceSlider.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    confidenceSlider.frameNStart = frameN  # exact frame index
                    confidenceSlider.tStart = t  # local t and not account for scr refresh
                    confidenceSlider.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(confidenceSlider, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    confidenceSlider.status = STARTED
                    confidenceSlider.setAutoDraw(True)
                
                # if confidenceSlider is active this frame...
                if confidenceSlider.status == STARTED:
                    # update params
                    pass
                
                # *buttonText* updates
                
                # if buttonText is starting this frame...
                if buttonText.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    buttonText.frameNStart = frameN  # exact frame index
                    buttonText.tStart = t  # local t and not account for scr refresh
                    buttonText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(buttonText, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    buttonText.status = STARTED
                    buttonText.setAutoDraw(True)
                
                # if buttonText is active this frame...
                if buttonText.status == STARTED:
                    # update params
                    pass
                
                # *confidenceProceedButton* updates
                
                # if confidenceProceedButton is starting this frame...
                if confidenceProceedButton.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    confidenceProceedButton.frameNStart = frameN  # exact frame index
                    confidenceProceedButton.tStart = t  # local t and not account for scr refresh
                    confidenceProceedButton.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(confidenceProceedButton, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    confidenceProceedButton.status = STARTED
                    # keyboard checking is just starting
                    confidenceProceedButton.clock.reset()  # now t=0
                if confidenceProceedButton.status == STARTED:
                    theseKeys = confidenceProceedButton.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                    _confidenceProceedButton_allKeys.extend(theseKeys)
                    if len(_confidenceProceedButton_allKeys):
                        confidenceProceedButton.keys = _confidenceProceedButton_allKeys[-1].name  # just the last key pressed
                        confidenceProceedButton.rt = _confidenceProceedButton_allKeys[-1].rt
                        confidenceProceedButton.duration = _confidenceProceedButton_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=confidence_evaluation,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    confidence_evaluation.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if confidence_evaluation.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in confidence_evaluation.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "confidence_evaluation" ---
            for thisComponent in confidence_evaluation.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for confidence_evaluation
            confidence_evaluation.tStop = globalClock.getTime(format='float')
            confidence_evaluation.tStopRefresh = tThisFlipGlobal
            # Run 'End Routine' code from skipConfidenceRoutine
            score = confidenceSlider.getRating()
            if score is None:
                score = -1
            
            confidenceScores[currentСonfIdea] = score
            
            thisExp.addData('conf_idea', currentСonfIdea)
            thisExp.addData('conf_score', score)
            confidence_loop.addData('confidenceSlider.response', confidenceSlider.getRating())
            # the Routine "confidence_evaluation" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisConfidence_loop as finished
            if hasattr(thisConfidence_loop, 'status'):
                thisConfidence_loop.status = FINISHED
            # if awaiting a pause, pause now
            if confidence_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                confidence_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed len(rememberedIdeas) repeats of 'confidence_loop'
        confidence_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if confidence_loop.trialList in ([], [None], None):
            params = []
        else:
            params = confidence_loop.trialList[0].keys()
        # save data for this loop
        confidence_loop.saveAsExcel(filename + '.xlsx', sheetName='confidence_loop',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "prepare_source_routine" ---
        # create an object to store info about Routine prepare_source_routine
        prepare_source_routine = data.Routine(
            name='prepare_source_routine',
            components=[],
        )
        prepare_source_routine.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from prepareSourceRoutine
        ideasForSource = [
            idea for idea, score in confidenceScores.items()
            if score != -1 and score < 50
        ]
        
        sourceRepsN = len(ideasForSource)
        
        continueRoutine = False
        # store start times for prepare_source_routine
        prepare_source_routine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prepare_source_routine.tStart = globalClock.getTime(format='float')
        prepare_source_routine.status = STARTED
        prepare_source_routine.maxDuration = None
        # keep track of which components have finished
        prepare_source_routineComponents = prepare_source_routine.components
        for thisComponent in prepare_source_routine.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prepare_source_routine" ---
        thisExp.currentRoutine = prepare_source_routine
        prepare_source_routine.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=prepare_source_routine,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                prepare_source_routine.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if prepare_source_routine.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in prepare_source_routine.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prepare_source_routine" ---
        for thisComponent in prepare_source_routine.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prepare_source_routine
        prepare_source_routine.tStop = globalClock.getTime(format='float')
        prepare_source_routine.tStopRefresh = tThisFlipGlobal
        # the Routine "prepare_source_routine" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        source_loop = data.TrialHandler2(
            name='source_loop',
            nReps=sourceRepsN, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(source_loop)  # add the loop to the experiment
        thisSource_loop = source_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisSource_loop.rgb)
        if thisSource_loop != None:
            for paramName in thisSource_loop:
                globals()[paramName] = thisSource_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisSource_loop in source_loop:
            source_loop.status = STARTED
            if hasattr(thisSource_loop, 'status'):
                thisSource_loop.status = STARTED
            currentLoop = source_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisSource_loop.rgb)
            if thisSource_loop != None:
                for paramName in thisSource_loop:
                    globals()[paramName] = thisSource_loop[paramName]
            
            # --- Prepare to start Routine "source_evaluation" ---
            # create an object to store info about Routine source_evaluation
            source_evaluation = data.Routine(
                name='source_evaluation',
                components=[sourceIdeaLabel, sourceEvaluationText, sourceEvaluationTextBox, sourceButtonText, sourceEvaluationButton],
            )
            source_evaluation.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            sourceIdeaLabel.reset()
            sourceEvaluationTextBox.reset()
            # create starting attributes for sourceEvaluationButton
            sourceEvaluationButton.keys = []
            sourceEvaluationButton.rt = []
            _sourceEvaluationButton_allKeys = []
            # Run 'Begin Routine' code from prepareIdeasForSource
            currentSourceIdea = ideasForSource[source_loop.thisN]
            sourceIdeaLabel.text = f'Идея: «{currentSourceIdea}»'
            # store start times for source_evaluation
            source_evaluation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            source_evaluation.tStart = globalClock.getTime(format='float')
            source_evaluation.status = STARTED
            source_evaluation.maxDuration = None
            # keep track of which components have finished
            source_evaluationComponents = source_evaluation.components
            for thisComponent in source_evaluation.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "source_evaluation" ---
            thisExp.currentRoutine = source_evaluation
            source_evaluation.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisSource_loop, 'status') and thisSource_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *sourceIdeaLabel* updates
                
                # if sourceIdeaLabel is starting this frame...
                if sourceIdeaLabel.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    sourceIdeaLabel.frameNStart = frameN  # exact frame index
                    sourceIdeaLabel.tStart = t  # local t and not account for scr refresh
                    sourceIdeaLabel.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sourceIdeaLabel, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    sourceIdeaLabel.status = STARTED
                    sourceIdeaLabel.setAutoDraw(True)
                
                # if sourceIdeaLabel is active this frame...
                if sourceIdeaLabel.status == STARTED:
                    # update params
                    pass
                
                # *sourceEvaluationText* updates
                
                # if sourceEvaluationText is starting this frame...
                if sourceEvaluationText.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    sourceEvaluationText.frameNStart = frameN  # exact frame index
                    sourceEvaluationText.tStart = t  # local t and not account for scr refresh
                    sourceEvaluationText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sourceEvaluationText, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    sourceEvaluationText.status = STARTED
                    sourceEvaluationText.setAutoDraw(True)
                
                # if sourceEvaluationText is active this frame...
                if sourceEvaluationText.status == STARTED:
                    # update params
                    pass
                
                # *sourceEvaluationTextBox* updates
                
                # if sourceEvaluationTextBox is starting this frame...
                if sourceEvaluationTextBox.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    sourceEvaluationTextBox.frameNStart = frameN  # exact frame index
                    sourceEvaluationTextBox.tStart = t  # local t and not account for scr refresh
                    sourceEvaluationTextBox.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sourceEvaluationTextBox, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    sourceEvaluationTextBox.status = STARTED
                    sourceEvaluationTextBox.setAutoDraw(True)
                
                # if sourceEvaluationTextBox is active this frame...
                if sourceEvaluationTextBox.status == STARTED:
                    # update params
                    pass
                
                # *sourceButtonText* updates
                
                # if sourceButtonText is starting this frame...
                if sourceButtonText.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    sourceButtonText.frameNStart = frameN  # exact frame index
                    sourceButtonText.tStart = t  # local t and not account for scr refresh
                    sourceButtonText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sourceButtonText, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    sourceButtonText.status = STARTED
                    sourceButtonText.setAutoDraw(True)
                
                # if sourceButtonText is active this frame...
                if sourceButtonText.status == STARTED:
                    # update params
                    pass
                
                # *sourceEvaluationButton* updates
                
                # if sourceEvaluationButton is starting this frame...
                if sourceEvaluationButton.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    sourceEvaluationButton.frameNStart = frameN  # exact frame index
                    sourceEvaluationButton.tStart = t  # local t and not account for scr refresh
                    sourceEvaluationButton.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sourceEvaluationButton, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    sourceEvaluationButton.status = STARTED
                    # keyboard checking is just starting
                    sourceEvaluationButton.clock.reset()  # now t=0
                if sourceEvaluationButton.status == STARTED:
                    theseKeys = sourceEvaluationButton.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                    _sourceEvaluationButton_allKeys.extend(theseKeys)
                    if len(_sourceEvaluationButton_allKeys):
                        sourceEvaluationButton.keys = _sourceEvaluationButton_allKeys[-1].name  # just the last key pressed
                        sourceEvaluationButton.rt = _sourceEvaluationButton_allKeys[-1].rt
                        sourceEvaluationButton.duration = _sourceEvaluationButton_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=source_evaluation,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    source_evaluation.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if source_evaluation.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in source_evaluation.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "source_evaluation" ---
            for thisComponent in source_evaluation.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for source_evaluation
            source_evaluation.tStop = globalClock.getTime(format='float')
            source_evaluation.tStopRefresh = tThisFlipGlobal
            source_loop.addData('sourceEvaluationTextBox.text',sourceEvaluationTextBox.text)
            # Run 'End Routine' code from formatSourceText
            sourceText = sourceEvaluationTextBox.getText()
            sourceText = sourceText.strip() if sourceText else ''
            
            sourceTexts[currentSourceIdea] = sourceText
            
            thisExp.addData('source_idea', currentSourceIdea)
            thisExp.addData('source_text', sourceText)
            # the Routine "source_evaluation" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisSource_loop as finished
            if hasattr(thisSource_loop, 'status'):
                thisSource_loop.status = FINISHED
            # if awaiting a pause, pause now
            if source_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                source_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed sourceRepsN repeats of 'source_loop'
        source_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if source_loop.trialList in ([], [None], None):
            params = []
        else:
            params = source_loop.trialList[0].keys()
        # save data for this loop
        source_loop.saveAsExcel(filename + '.xlsx', sheetName='source_loop',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed totalIdeas repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if trials.trialList in ([], [None], None):
        params = []
    else:
        params = trials.trialList[0].keys()
    # save data for this loop
    trials.saveAsExcel(filename + '.xlsx', sheetName='trials',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "thank_you" ---
    # create an object to store info about Routine thank_you
    thank_you = data.Routine(
        name='thank_you',
        components=[thankYouText],
    )
    thank_you.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for thank_you
    thank_you.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    thank_you.tStart = globalClock.getTime(format='float')
    thank_you.status = STARTED
    thank_you.maxDuration = None
    # keep track of which components have finished
    thank_youComponents = thank_you.components
    for thisComponent in thank_you.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thank_you" ---
    thisExp.currentRoutine = thank_you
    thank_you.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thankYouText* updates
        
        # if thankYouText is starting this frame...
        if thankYouText.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thankYouText.frameNStart = frameN  # exact frame index
            thankYouText.tStart = t  # local t and not account for scr refresh
            thankYouText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thankYouText, 'tStartRefresh')  # time at next scr refresh
            # update status
            thankYouText.status = STARTED
            thankYouText.setAutoDraw(True)
        
        # if thankYouText is active this frame...
        if thankYouText.status == STARTED:
            # update params
            pass
        
        # if thankYouText is stopping this frame...
        if thankYouText.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thankYouText.tStartRefresh + 4-frameTolerance:
                # keep track of stop time/frame for later
                thankYouText.tStop = t  # not accounting for scr refresh
                thankYouText.tStopRefresh = tThisFlipGlobal  # on global time
                thankYouText.frameNStop = frameN  # exact frame index
                # update status
                thankYouText.status = FINISHED
                thankYouText.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=thank_you,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            thank_you.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if thank_you.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in thank_you.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thank_you" ---
    for thisComponent in thank_you.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for thank_you
    thank_you.tStop = globalClock.getTime(format='float')
    thank_you.tStopRefresh = tThisFlipGlobal
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if thank_you.maxDurationReached:
        routineTimer.addTime(-thank_you.maxDuration)
    elif thank_you.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)
    # end 'rush' mode
    core.rush(enable=False)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
