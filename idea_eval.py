#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.0),
    on March 22, 2026, at 19:12
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
    'participantID': ["A800", "A470", "B870"],
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
        originPath='C:\\Users\\aivan\\Documents\\source_monitor\\idea_eval.py',
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
    
    # --- Initialize components for Routine "instructions" ---
    # Run 'Begin Experiment' code from initialiseConditions
    skipConfidence = False
    skipSource = False
    ceInstructionCount = 0
    # Run 'Begin Experiment' code from getParticipantId
    participantId = expInfo['participantID']
    stimuliPath = f'./stimuli/{participantId}_ideas.csv'
    inastructionText = visual.TextStim(win=win, name='inastructionText',
        text='Сейчас вам будет показан список идей.\n\nДля каждой идеи ответьте на вопрос:  помните ли вы, что эта идея была озвучена  кем-либо из участников во время сессии?\n\nЕсли помните — вы оцените свою уверенность  в том, кто именно был автором идеи. \n\nДля начала нажмите ПРОБЕЛ',
        font='Courier New',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    proceedKey = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "routine_instructions" ---
    routineInstructionsText = visual.TextStim(win=win, name='routineInstructionsText',
        text='Ровно неделю назад Вы предлагали идеи необычного использования для следующего предмета:',
        font='Courier New',
        pos=(0, 0.15), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    coreItemText = visual.TextStim(win=win, name='coreItemText',
        text='',
        font='Courier New',
        pos=(0, 0), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "show_idea" ---
    showIdeaText = visual.TextStim(win=win, name='showIdeaText',
        text='',
        font='Courier New',
        pos=(0, 0.15), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    ideaQuestionText = visual.TextStim(win=win, name='ideaQuestionText',
        text='Вы помните, что эта идея была озвучена?\n\n[Y] — да     [N] — нет',
        font='Courier New',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    showIdeaProceedButton = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "confidence_evaluation_instruction" ---
    ceInstructionText = visual.TextStim(win=win, name='ceInstructionText',
        text='Вам будет предложено оценить, помните ли вы идею\n\nЕсли помните — вы оцените свою уверенность \nв том, кто именно был автором идеи. ',
        font='Courier New',
        pos=(0, 0.21), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    ceInstructionHint = visual.TextStim(win=win, name='ceInstructionHint',
        text='Нажмите [Д] — «да, помню»\nНажмите [Н] — «нет, не помню»',
        font='Courier New',
        pos=(0, -0.1), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    ceInstructionButtonText = visual.TextStim(win=win, name='ceInstructionButtonText',
        text='Чтобы продолжить, нажмите ПРОБЕЛ',
        font='Courier New',
        pos=(0, -0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    ceInstructionProceedButton = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "confidence_evaluation" ---
    confidenceText = visual.TextStim(win=win, name='confidenceText',
        text='Насколько вы уверены, что эту идею предложил именно партнёр, а не она была заимствована из другой идеи?',
        font='Courier New',
        pos=(0, 0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    confidenceTextHint = visual.TextStim(win=win, name='confidenceTextHint',
        text='0% — точно заимствована     100% — точно его/её идея',
        font='Courier New',
        pos=(0, -0.07), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    confidenceSlider = visual.Slider(win=win, name='confidenceSlider',
        startValue=None, size=(1.0, 0.06), pos=(0, -0.25), units=win.units,
        labels=('0%', '50%', '100%'), ticks=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100), granularity=0.0,
        style='rating', styleTweaks=[], opacity=None,
        labelColor='LightGray', markerColor=(-0.4353, -0.5216, 0.0902), lineColor=(0.6000, 0.6941, 0.8510), colorSpace='rgb',
        font='Courier New', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    buttonText = visual.TextStim(win=win, name='buttonText',
        text='Нажмите ENTER для подтверждения',
        font='Courier New',
        pos=(0, -0.4), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    confidenceProceedButton = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "source_evaluation" ---
    sourceEvaluationText = visual.TextStim(win=win, name='sourceEvaluationText',
        text='Вы указали, что эта идея, вероятно, была заимствована.\n\nОпишите, из какой идеи, по вашему мнению, она была взята:',
        font='Courier New',
        pos=(0, 0.25), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    sourceEvaluationTextBox = visual.TextBox2(
         win, text=None, placeholder='Введите ваш ответ здесь...', font='Courier New',
         ori=0.0, pos=(0, -0.15), draggable=False,      letterHeight=0.05,
         size=(1.4, 0.2), borderWidth=1.0,
         color='black', colorSpace='rgb',
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
        pos=(0, -0.4), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    sourceEvaluationButton = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "save_routine" ---
    
    # --- Initialize components for Routine "thank_you" ---
    thankYouText = visual.TextStim(win=win, name='thankYouText',
        text='Спасибо, участник))))',
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
    
    # --- Prepare to start Routine "instructions" ---
    # create an object to store info about Routine instructions
    instructions = data.Routine(
        name='instructions',
        components=[inastructionText, proceedKey],
    )
    instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for proceedKey
    proceedKey.keys = []
    proceedKey.rt = []
    _proceedKey_allKeys = []
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
        
        # *proceedKey* updates
        
        # if proceedKey is starting this frame...
        if proceedKey.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            proceedKey.frameNStart = frameN  # exact frame index
            proceedKey.tStart = t  # local t and not account for scr refresh
            proceedKey.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(proceedKey, 'tStartRefresh')  # time at next scr refresh
            # update status
            proceedKey.status = STARTED
            # keyboard checking is just starting
            proceedKey.clock.reset()  # now t=0
        if proceedKey.status == STARTED:
            theseKeys = proceedKey.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _proceedKey_allKeys.extend(theseKeys)
            if len(_proceedKey_allKeys):
                proceedKey.keys = _proceedKey_allKeys[-1].name  # just the last key pressed
                proceedKey.rt = _proceedKey_allKeys[-1].rt
                proceedKey.duration = _proceedKey_allKeys[-1].duration
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
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(stimuliPath), 
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
            components=[routineInstructionsText, coreItemText],
        )
        routine_instructions.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from onCoreItemChange
        coreItem
        coreItemText.setText(coreIdea)
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
        while continueRoutine and routineTimer.getTime() < 5.0:
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
            
            # if routineInstructionsText is stopping this frame...
            if routineInstructionsText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > routineInstructionsText.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    routineInstructionsText.tStop = t  # not accounting for scr refresh
                    routineInstructionsText.tStopRefresh = tThisFlipGlobal  # on global time
                    routineInstructionsText.frameNStop = frameN  # exact frame index
                    # update status
                    routineInstructionsText.status = FINISHED
                    routineInstructionsText.setAutoDraw(False)
            
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
            
            # if coreItemText is stopping this frame...
            if coreItemText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > coreItemText.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    coreItemText.tStop = t  # not accounting for scr refresh
                    coreItemText.tStopRefresh = tThisFlipGlobal  # on global time
                    coreItemText.frameNStop = frameN  # exact frame index
                    # update status
                    coreItemText.status = FINISHED
                    coreItemText.setAutoDraw(False)
            
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
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routine_instructions.maxDurationReached:
            routineTimer.addTime(-routine_instructions.maxDuration)
        elif routine_instructions.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.000000)
        
        # --- Prepare to start Routine "show_idea" ---
        # create an object to store info about Routine show_idea
        show_idea = data.Routine(
            name='show_idea',
            components=[showIdeaText, ideaQuestionText, showIdeaProceedButton],
        )
        show_idea.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        showIdeaText.setText(ideaText)
        # create starting attributes for showIdeaProceedButton
        showIdeaProceedButton.keys = []
        showIdeaProceedButton.rt = []
        _showIdeaProceedButton_allKeys = []
        # store start times for show_idea
        show_idea.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        show_idea.tStart = globalClock.getTime(format='float')
        show_idea.status = STARTED
        show_idea.maxDuration = None
        # keep track of which components have finished
        show_ideaComponents = show_idea.components
        for thisComponent in show_idea.components:
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
        
        # --- Run Routine "show_idea" ---
        thisExp.currentRoutine = show_idea
        show_idea.forceEnded = routineForceEnded = not continueRoutine
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
            
            # *showIdeaText* updates
            
            # if showIdeaText is starting this frame...
            if showIdeaText.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                showIdeaText.frameNStart = frameN  # exact frame index
                showIdeaText.tStart = t  # local t and not account for scr refresh
                showIdeaText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(showIdeaText, 'tStartRefresh')  # time at next scr refresh
                # update status
                showIdeaText.status = STARTED
                showIdeaText.setAutoDraw(True)
            
            # if showIdeaText is active this frame...
            if showIdeaText.status == STARTED:
                # update params
                pass
            
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
                    currentRoutine=show_idea,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                show_idea.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if show_idea.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in show_idea.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "show_idea" ---
        for thisComponent in show_idea.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for show_idea
        show_idea.tStop = globalClock.getTime(format='float')
        show_idea.tStopRefresh = tThisFlipGlobal
        # check responses
        if showIdeaProceedButton.keys in ['', [], None]:  # No response was made
            showIdeaProceedButton.keys = None
        trials.addData('showIdeaProceedButton.keys',showIdeaProceedButton.keys)
        if showIdeaProceedButton.keys != None:  # we had a response
            trials.addData('showIdeaProceedButton.rt', showIdeaProceedButton.rt)
            trials.addData('showIdeaProceedButton.duration', showIdeaProceedButton.duration)
        # Run 'End Routine' code from storeResponseKey
        response = showIdeaProceedButton.keys
        if response == 'n':
            skipConfidence = True
            skipSource = True
        else:
            skipConfidence = False
            skipSource = False
        
        print(skipConfidence, skipSource, response)
        
        thisExp.addData('remembered', response)
        # the Routine "show_idea" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "confidence_evaluation_instruction" ---
        # create an object to store info about Routine confidence_evaluation_instruction
        confidence_evaluation_instruction = data.Routine(
            name='confidence_evaluation_instruction',
            components=[ceInstructionText, ceInstructionHint, ceInstructionButtonText, ceInstructionProceedButton],
        )
        confidence_evaluation_instruction.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from showOnce
        if ceInstructionCount >= 1 or skipConfidence or skipSource:
            continueRoutine = False
        else: 
            ceInstructionCount += 1
        # create starting attributes for ceInstructionProceedButton
        ceInstructionProceedButton.keys = []
        ceInstructionProceedButton.rt = []
        _ceInstructionProceedButton_allKeys = []
        # store start times for confidence_evaluation_instruction
        confidence_evaluation_instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        confidence_evaluation_instruction.tStart = globalClock.getTime(format='float')
        confidence_evaluation_instruction.status = STARTED
        confidence_evaluation_instruction.maxDuration = None
        # keep track of which components have finished
        confidence_evaluation_instructionComponents = confidence_evaluation_instruction.components
        for thisComponent in confidence_evaluation_instruction.components:
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
        
        # --- Run Routine "confidence_evaluation_instruction" ---
        thisExp.currentRoutine = confidence_evaluation_instruction
        confidence_evaluation_instruction.forceEnded = routineForceEnded = not continueRoutine
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
            
            # *ceInstructionText* updates
            
            # if ceInstructionText is starting this frame...
            if ceInstructionText.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ceInstructionText.frameNStart = frameN  # exact frame index
                ceInstructionText.tStart = t  # local t and not account for scr refresh
                ceInstructionText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ceInstructionText, 'tStartRefresh')  # time at next scr refresh
                # update status
                ceInstructionText.status = STARTED
                ceInstructionText.setAutoDraw(True)
            
            # if ceInstructionText is active this frame...
            if ceInstructionText.status == STARTED:
                # update params
                pass
            
            # *ceInstructionHint* updates
            
            # if ceInstructionHint is starting this frame...
            if ceInstructionHint.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ceInstructionHint.frameNStart = frameN  # exact frame index
                ceInstructionHint.tStart = t  # local t and not account for scr refresh
                ceInstructionHint.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ceInstructionHint, 'tStartRefresh')  # time at next scr refresh
                # update status
                ceInstructionHint.status = STARTED
                ceInstructionHint.setAutoDraw(True)
            
            # if ceInstructionHint is active this frame...
            if ceInstructionHint.status == STARTED:
                # update params
                pass
            
            # *ceInstructionButtonText* updates
            
            # if ceInstructionButtonText is starting this frame...
            if ceInstructionButtonText.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ceInstructionButtonText.frameNStart = frameN  # exact frame index
                ceInstructionButtonText.tStart = t  # local t and not account for scr refresh
                ceInstructionButtonText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ceInstructionButtonText, 'tStartRefresh')  # time at next scr refresh
                # update status
                ceInstructionButtonText.status = STARTED
                ceInstructionButtonText.setAutoDraw(True)
            
            # if ceInstructionButtonText is active this frame...
            if ceInstructionButtonText.status == STARTED:
                # update params
                pass
            
            # *ceInstructionProceedButton* updates
            waitOnFlip = False
            
            # if ceInstructionProceedButton is starting this frame...
            if ceInstructionProceedButton.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ceInstructionProceedButton.frameNStart = frameN  # exact frame index
                ceInstructionProceedButton.tStart = t  # local t and not account for scr refresh
                ceInstructionProceedButton.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ceInstructionProceedButton, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ceInstructionProceedButton.started')
                # update status
                ceInstructionProceedButton.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(ceInstructionProceedButton.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(ceInstructionProceedButton.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if ceInstructionProceedButton.status == STARTED and not waitOnFlip:
                theseKeys = ceInstructionProceedButton.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _ceInstructionProceedButton_allKeys.extend(theseKeys)
                if len(_ceInstructionProceedButton_allKeys):
                    ceInstructionProceedButton.keys = _ceInstructionProceedButton_allKeys[-1].name  # just the last key pressed
                    ceInstructionProceedButton.rt = _ceInstructionProceedButton_allKeys[-1].rt
                    ceInstructionProceedButton.duration = _ceInstructionProceedButton_allKeys[-1].duration
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
                    currentRoutine=confidence_evaluation_instruction,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                confidence_evaluation_instruction.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if confidence_evaluation_instruction.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in confidence_evaluation_instruction.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "confidence_evaluation_instruction" ---
        for thisComponent in confidence_evaluation_instruction.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for confidence_evaluation_instruction
        confidence_evaluation_instruction.tStop = globalClock.getTime(format='float')
        confidence_evaluation_instruction.tStopRefresh = tThisFlipGlobal
        # check responses
        if ceInstructionProceedButton.keys in ['', [], None]:  # No response was made
            ceInstructionProceedButton.keys = None
        trials.addData('ceInstructionProceedButton.keys',ceInstructionProceedButton.keys)
        if ceInstructionProceedButton.keys != None:  # we had a response
            trials.addData('ceInstructionProceedButton.rt', ceInstructionProceedButton.rt)
            trials.addData('ceInstructionProceedButton.duration', ceInstructionProceedButton.duration)
        # the Routine "confidence_evaluation_instruction" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "confidence_evaluation" ---
        # create an object to store info about Routine confidence_evaluation
        confidence_evaluation = data.Routine(
            name='confidence_evaluation',
            components=[confidenceText, confidenceTextHint, confidenceSlider, buttonText, confidenceProceedButton],
        )
        confidence_evaluation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from skipConfidenceRoutine
        if skipConfidence:
            continueRoutine = False
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
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
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
            if confidenceSlider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                confidenceSlider.frameNStart = frameN  # exact frame index
                confidenceSlider.tStart = t  # local t and not account for scr refresh
                confidenceSlider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(confidenceSlider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'confidenceSlider.started')
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
        trials.addData('confidenceSlider.response', confidenceSlider.getRating())
        trials.addData('confidenceSlider.rt', confidenceSlider.getRT())
        # Run 'End Routine' code from skipSourceRoutine
        confidenceValue = confidenceSlider.getRating()
        
        if confidenceValue is None:
            confidenceValue = -1
        
        thisExp.addData('confidence', confidenceValue)
        
        if confidenceValue != -1 and confidenceValue < 50:
            skipSource = False
        else:
            skipSource = True
        # the Routine "confidence_evaluation" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "source_evaluation" ---
        # create an object to store info about Routine source_evaluation
        source_evaluation = data.Routine(
            name='source_evaluation',
            components=[sourceEvaluationText, sourceEvaluationTextBox, sourceButtonText, sourceEvaluationButton],
        )
        source_evaluation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from skipSourceEvaluation
        if skipSource:
            continueRoutine = False
        sourceEvaluationTextBox.reset()
        # create starting attributes for sourceEvaluationButton
        sourceEvaluationButton.keys = []
        sourceEvaluationButton.rt = []
        _sourceEvaluationButton_allKeys = []
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
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
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
            waitOnFlip = False
            
            # if sourceEvaluationButton is starting this frame...
            if sourceEvaluationButton.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sourceEvaluationButton.frameNStart = frameN  # exact frame index
                sourceEvaluationButton.tStart = t  # local t and not account for scr refresh
                sourceEvaluationButton.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sourceEvaluationButton, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sourceEvaluationButton.started')
                # update status
                sourceEvaluationButton.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(sourceEvaluationButton.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(sourceEvaluationButton.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if sourceEvaluationButton.status == STARTED and not waitOnFlip:
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
        trials.addData('sourceEvaluationTextBox.text',sourceEvaluationTextBox.text)
        # check responses
        if sourceEvaluationButton.keys in ['', [], None]:  # No response was made
            sourceEvaluationButton.keys = None
        trials.addData('sourceEvaluationButton.keys',sourceEvaluationButton.keys)
        if sourceEvaluationButton.keys != None:  # we had a response
            trials.addData('sourceEvaluationButton.rt', sourceEvaluationButton.rt)
            trials.addData('sourceEvaluationButton.duration', sourceEvaluationButton.duration)
        # Run 'End Routine' code from formatSourceText
        sourceText = sourceEvaluationTextBox.getText()
        
        sourceText = sourceText.strip() if sourceText else ''
        
        thisExp.addData('source_description', sourceText)
        # the Routine "source_evaluation" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "save_routine" ---
        # create an object to store info about Routine save_routine
        save_routine = data.Routine(
            name='save_routine',
            components=[],
        )
        save_routine.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from saveTableRows
        if skipConfidence:
            thisExp.addData('confidence', '')
        if skipSource:
            thisExp.addData('source_description', '')
        
        continueRoutine = False
        # store start times for save_routine
        save_routine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        save_routine.tStart = globalClock.getTime(format='float')
        save_routine.status = STARTED
        save_routine.maxDuration = None
        # keep track of which components have finished
        save_routineComponents = save_routine.components
        for thisComponent in save_routine.components:
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
        
        # --- Run Routine "save_routine" ---
        thisExp.currentRoutine = save_routine
        save_routine.forceEnded = routineForceEnded = not continueRoutine
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
                    currentRoutine=save_routine,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                save_routine.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if save_routine.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in save_routine.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "save_routine" ---
        for thisComponent in save_routine.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for save_routine
        save_routine.tStop = globalClock.getTime(format='float')
        save_routine.tStopRefresh = tThisFlipGlobal
        # the Routine "save_routine" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
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
        
    # completed 1 repeats of 'trials'
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
