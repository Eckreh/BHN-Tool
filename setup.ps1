# Check if the "venv" folder exists in the current directory
if (-not (Test-Path venv)) {
    # If the folder does not exist, run the command to create the virtual environment
    python -m venv venv
    Write-Host "Virtual environment 'venv' has been created."
    
    # Activate the virtual environment
    $activateScriptPath = ".\venv\Scripts\Activate.ps1"
    if (Test-Path $activateScriptPath) {
        # If the activate script exists, run it
        . $activateScriptPath
        Write-Host "Virtual environment 'venv' has been activated."

        # Install requirements
        if (Test-Path "requirements.txt") {
            # If requirements.txt exists, install the requirements using pip
            pip install -r requirements.txt
            Write-Host "Requirements installed."
        } else {
            Write-Host "requirements.txt not found. Skipping installation."
        }
    } else {
        Write-Host "Activate script not found. Virtual environment activation failed."
    }
} else {
    # If the folder exists, activate the virtual environment
    $activateScriptPath = ".\venv\Scripts\Activate.ps1"
    if (Test-Path $activateScriptPath) {
        # If the activate script exists, run it
        . $activateScriptPath
        Write-Host "Virtual environment 'venv' has been activated."
    } else {
        Write-Host "Activate script not found. Virtual environment activation failed."
    }
}
