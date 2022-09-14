"""
Description:
Automation class.

Requirements:
this pipeline class, needs to have the following methods:
    connect
    disconnect
    is_connected
    decision_mapping
    send_action
"""
from ipaddress import ip_address
import logging
import csv
from typing import List, Tuple
from pymodbus.client.sync import ModbusTcpClient
from time import time



class AutomationClass():

    # get a logger
    logger = logging.getLogger()
    client = None

    ip_address = '10.10.11.110'
    port = '32700'
    
    start_flag_addr = 0
    mark_flag_addr = 1

    mark_peeling = 1
    mark_white = 1
    mark_silver = 1
    mark_scuff = 1

    
    line_speed_addr = 2
    line_speed_val = 5

    decision_addr_0 = 3
    decision_addr_1 = 4
    decision_addr_2 = 5

    handshake_addr_0 = 6
    handshake_val_0 = 0
    handshake_addr_1 = 7
    handshake_val_1 = 0
    handshake_addr_2 = 8
    handshake_val_2 = 0

   
    def __init__(self, ip_address, port) -> None:
       self.ip_address = ip_address
       self.port = port

    def connect(self) -> None:
        self.logger.info("connecting")
        try:
            self.client = ModbusTcpClient(self.ip_address, self.port)
            self.client.connect()

            if self.is_connected():
                self.logger.info("connected to plc")
                result = self.client.write_register(self.start_flag_addr, 1)
                if result.isError():
                    self.logger.error("Unable to set start flag")
                self.initilize_plc()
                
        except Exception as e:
            print(e)
            self.logger.error(e)

    def disconnect(self) -> None:
        self.logger.info("disconnecting")
        try:
            if self.is_connected():
                result = self.client.write_register(self.start_flag_addr, 0)
                if result.isError():
                    self.logger.error("Unable to set start flag")
            self.client.close()
            
        except Exception as ex:
            self.logger.error(ex)

    def is_connected(self) -> bool:
        self.logger.debug("is_connected")
        try:
            if self.client is not None:
                return self.client.is_socket_open()
            else:
                return True
        except Exception as e:
            self.logger.error(e)
        return True
    
    def decision_mapping(self, msg: str) -> Tuple[str, List[str]]:
        self.logger.debug("decision_mapping")
        
        decisions = msg['decision'].split(",")
        if self.mark_scuff == 1 and 'scuff' in decisions:
            return "mark", [msg['sensor_topic']]
        elif self.mark_peeling == 1 and 'peeling' in decisions:
            return "mark", [msg['sensor_topic']]
        elif self.mark_silver == 1 and 'silver' in decisions:
            return "mark", [msg['sensor_topic']]
        elif self.mark_white == 1 and 'white' in decisions:
            return "mark", [msg['sensor_topic']]
        else:
            return "no mark", [msg['sensor_topic'], self.user_ID, self.batch_ID]

    def send_action(self, action: str, aux_info: List[str]) -> bool:
        self.logger.info(f"send_action: {action}")
        sensor_topic = aux_info[0]

        if self.client is None:
            self.logger.info("Client not initilized")
            return True        

        if sensor_topic == 'sensor/gadget-sensor-avt/0':
            addr = self.decision_addr_0
        elif sensor_topic == 'sensor/gadget-sensor-avt/1':
            addr = self.decision_addr_1
        elif sensor_topic == 'sensor/gadget-sensor-avt/2':
            addr = self.decision_addr_2

        try:
            if action == "mark":
                result = self.client.write_register(addr, 1)
                if result.isError():
                    self.logger.error(result) 
            else:
                result = self.client.write_register(addr, 0)
                if result.isError():
                    self.logger.error(result) 
                
        except Exception as e:
            self.logger.error(e)
        
        self.handshake(sensor_topic)

    def mark_peeling_handler(self, mark_peeling):
        self.logger.info(f"Updating mark_peeling to {mark_peeling}")
        self.mark_peeling = mark_peeling

    def mark_white_handler(self, mark_white):
        self.logger.info(f"Updating mark_peeling to {mark_white}")
        self.mark_white = mark_white

    def mark_silver_handler(self, mark_silver):
        self.logger.info(f"Updating mark_peeling to {mark_silver}")
        self.mark_silver = mark_silver

    def mark_scuff_handler(self, mark_scuff):
        self.logger.info(f"Updating mark_peeling to {mark_scuff}")
        self.mark_scuff = mark_scuff

    def line_speed_handler(self, line_speed: int) -> None:
        self.logger.info(f"Updating line speed to {line_speed}")
        try:
            self.line_speed_val = line_speed
            result = self.client.write_register(self.line_speed_addr, int((line_speed * 10)))
            if result.isError():
                self.logger.error("Unable to set line speed")
        except Exception as ex:
            self.logger.error(ex)
    
    def user_id_handler(self, user_ID: str) -> None:
        self.logger.info(f"Update user id to {user_ID}")
        self.user_ID = user_ID

    def batch_id_handler(self, batch_ID: str) -> None:
        self.logger.info(f"Update batch id to {batch_ID}")
        self.batch_ID = batch_ID


    
if __name__ == '__main__':
    auto = AutomationClass(0, 0)
    auto.connect()
    print(f"Is connected? {auto.is_connected()}")
    auto.start_flag_handler(start_flag=1)
    auto.mark_peeling_handler(mark_peeling=1)
    auto.line_speed_handler(line_speed = 5)

    print("done")
    auto.client.close()
