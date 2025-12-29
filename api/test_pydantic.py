import sys
from config.models import TrainingConfig
from pydantic import ValidationError

print('Testing Pydantic validation...')
try:
    config = TrainingConfig(target_feature='Total_Score', train_split=0.8)
    print('✅ Valid config accepted')
except Exception as e:
    print(f'❌ Error: {e}')
print('✅ Tests complete!')
