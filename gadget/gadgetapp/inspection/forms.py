# Use forms for user inputs
from tkinter import Widget
from django import forms

from inspection.models import UserSensorSelector, UserSensorConfig, UserAutomationConfig, UserPipelineConfig

class ChangePipelineForm(forms.ModelForm):
    class Meta:
        model = UserPipelineConfig
        fields = (
            'confidence_peeling',
            'confidence_catheter',
            'confidence_white',
            'confidence_scuff',
        )

        labels = {
            'confidence_peeling': 'Peeling Confidence',
            'confidence_catheter': 'Catheter Confidence',
            'confidence_white':'White Confidence',
            'confidence_scuff':'Scuff Confidence',
        }

        widgets = {
            'confidence_peeling': forms.TextInput(attrs={'class': 'config-input_field'}),
            'confidence_catheter': forms.TextInput(attrs={'class': 'config-input_field'}),
            'confidence_white': forms.TextInput(attrs={'class': 'config-input_field'}),
            'confidence_scuff': forms.TextInput(attrs={'class': 'config-input_field'}),
        }

class ChooseSensorForm(forms.ModelForm):
    # Meta tells Django what database elements/fields to reference
    class Meta:
        # database element
        model = UserSensorSelector
        # exposed fields
        fields=('current_sensor',)
        labels={'current_sensor':'Available Sensors'}
        widgets = {'current_sensor': forms.Select(attrs={'class': 'config-input_field'}),}

class ChangeSensorForm(forms.ModelForm):
    
    class Meta:
        trigger_type_choices = [(False, 'External'), (True, 'Time Based')]
        model=UserSensorConfig
        fields=('timed_trigger_mode','gain','exposure_time',)
        labels={'timed_trigger_mode':'Trigger Mode','gain':'Gain','exposure_time':'Exposure Time'}
        widgets={
            'timed_trigger_mode': forms.Select(choices=trigger_type_choices, attrs={'class': 'config-input_field'}),
            'gain': forms.NumberInput(attrs={'class': 'config-input_field'}),
            'exposure': forms.NumberInput(attrs={'class': 'config-input_field'}),
            }
  
class ChangeAutomationForm(forms.ModelForm):
    class Meta:
        model = UserAutomationConfig
        fields = ('user_ID','batch_ID','line_speed','mark_peeling','mark_white','mark_scuff')
        labels = {
            'user_ID':'User ID',
            'batch_ID':'Batch ID',
            'line_speed':'Line Speed (units)',
            'mark_peeling':'Mark Peeling',
            'mark_white':'Mark White',
            'mark_scuff':'Mark Scuff',
        }
        CHOICES = [(True, 'YES'), (False, 'NO')]
        widgets = {
            'user_ID': forms.TextInput(attrs={'class': 'config-input_field'}),
            'batch_ID': forms.TextInput(attrs={'class': 'config-input_field'}),
            'line_speed': forms.NumberInput(attrs={'class': 'config-input_field'}),
            'mark_peeling': forms.RadioSelect(choices=CHOICES),
            'mark_white': forms.RadioSelect(choices=CHOICES),
            'mark_scuff': forms.RadioSelect(choices=CHOICES),
        }

