
"""Hybrid Rule + ML Tip Selector"""

class HybridRuleMLSystem:
    def __init__(self, mood_detector, tip_generator):
        self.mood_detector = mood_detector
        self.tip_generator = tip_generator
        self.rule_db = {
            'tired': ['Take rest', 'Sleep early'],
            'anxious': ['Breathe deeply', 'Write down worries'],
            'sad': ['Talk to a friend', 'Practice gratitude'],
            'neutral': ['Set small goals', 'Reflect calmly'],
            'hopeful': ['Visualize goals', 'Plan next steps'],
            'happy': ['Celebrate wins', 'Share positivity']
        }
    def get_tips(self, mood, confidence=0.8):
        if confidence < 0.7 and mood in self.rule_db:
            return self.rule_db[mood]
        return self.tip_generator.generate_tips(mood, num_tips=3)
