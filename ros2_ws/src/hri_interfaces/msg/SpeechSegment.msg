# msg/SpeechSegment.msg
# Frase completa, publicada al detectarse el fin del habla. Lleva el audio ya
# acumulado para que transcripción y verificación de voz no repitan el
# buffering cada una por su cuenta.
# header.stamp marca el inicio del segmento; el final se deriva de
# samples.size() / sample_rate.
std_msgs/Header header
float32[] samples
uint32 sample_rate
uint8 channels
