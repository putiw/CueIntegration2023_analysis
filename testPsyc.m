% Clear the workspace and the screen
sca;
close all;
clearvars;

% Set up the Psychtoolbox screen
PsychDefaultSetup(2);
Screen('Preference', 'SkipSyncTests', 1);
screens = Screen('Screens');
screenNumber = max(screens);
white = WhiteIndex(screenNumber);
black = BlackIndex(screenNumber);
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, white, [], [], [], 4);
[xCenter, yCenter] = RectCenter(windowRect);
Screen('BlendFunction', window, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

% Set up the stimulus parameters
dotSize = 20;
dotRadius = 10; % in cm
nDots = 20;
coherence = 0.8;
speed = 5; % in cm/s
duration = 3; % in seconds
gap = 6; % in seconds
numTrials = 10;
trialDirections = [ones(numTrials/2,1); -ones(numTrials/2,1)];
trialOrder = Shuffle(trialDirections);

% Set up the dots
dots = rand(2, nDots) * 2 - 1; % random positions within a square
distFromCenter = sqrt(sum(dots.^2, 1));
dots(:, distFromCenter > dotRadius) = NaN; % keep only dots within the circle
cohDots = randsample(nDots, round(coherence * nDots)); % coherent dots

% Set up the response keys
upKey = KbName('UpArrow');
downKey = KbName('DownArrow');

% Start the experiment
for trial = 1:numTrials
    % Randomly assign the motion direction
    direction = trialOrder(trial);
    cohDots(1:round((1-abs(direction))*nDots)) = [];
    if direction == -1
        speed = -speed;
    end
    
    % Set up the dots for this trial
    dotsThisTrial = dots;
    dotsThisTrial(:,cohDots) = repmat(speed * [1; 0], 1, length(cohDots));
    
    % Start the trial
    Screen('FillRect', window, black);
    Screen('Flip', window);
    WaitSecs(gap);
    for t = 1:duration*60 % 60 Hz refresh rate
        % Update the dots
        dotsThisTrial = dotsThisTrial + speed/60 * [cosd(45) -sind(45); sind(45) cosd(45)] * dotsThisTrial;
        distFromCenter = sqrt(sum(dotsThisTrial.^2, 1));
        dotsThisTrial(:, distFromCenter > dotRadius) = NaN;
        cohDots = find(~isnan(dotsThisTrial(1,:)));
        
        % Draw the dots
        Screen('SelectStereoDrawBuffer', window, 0);
        Screen('DrawDots', window, dotsThisTrial, dotSize, white,
        
        sca
        
        
        sca
        ççsca
        sca
        sca
        
        
        sca
        
        
        c[xCenter/2 yCenter], 1);
        Screen('SelectStereoDrawBuffer', window, 1);
        Screen('DrawDots', window, dotsThisTrial, dotSize, white, [xCenter*3/2 yCenter], 1);
        
        % Flip the screen
        vbl = Screen('Flip', window, [], 1);
        
               % Check for responses during the blank gap
        if t == 1
            respToBeMade = true;
            while respToBeMade && (GetSecs - gap) < duration
                [keyIsDown, secs, keyCode] = KbCheck;
                if keyIsDown
                    if keyCode(upKey)
                        response(trial) = 1;
                        respToBeMade = false;
                    elseif keyCode(downKey)
                        response(trial) = -1;
                        respToBeMade = false;
                    end
                end
            end
        end
    end
    
    % End the trial
    Screen('FillRect', window, black);
    Screen('Flip', window);
    WaitSecs(gap);
end

% Close the screen and report the responses
sca;
disp(response);
