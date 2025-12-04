# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"

import json
import os
import re
from io import BytesIO
import subprocess
from subprocess import CompletedProcess
import platform
import time
from typing import (
    Optional,
    Union
)
import uuid
from PIL import Image
from hmdriver2.driver import Driver

from utils import (
    encode_image,
    print_out
)


class Operate(object):
    def __init__(self,
                 bundle_name_dict: dict,
                 hdc_command: str = "hdc.exe",
                 factor: float = 0.5) -> None:
        self.bundle_name_dict = bundle_name_dict
        self.hdc_command = hdc_command
        self.factor = factor
        self.device_id = self._get_device_id()
        self.driver = Driver(serial=self.device_id)
    
    def _get_device_id(self) -> str:
        devices = self.get_connected_devices()
        device_id = ""
        length = len(devices)

        if length > 1:
            print_out(
                "please to choose which devices id:",
                stdout=True
            )
            for idx, sub_device in enumerate(devices):
                print_out(
                    f"{idx + 1}. {sub_device}",
                    stdout=True
                )
            while True:
                choice = input("your choice (input index): ")
                print_out(f"your choice (input index): {choice}")
                if not choice.isdigit():
                    print_out(
                        "Invalid input, please enter a number.",
                        stdout=True
                    )
                    continue
                cur_idx = int(choice) - 1
                if 0 <= cur_idx < length:
                    device_id = devices[cur_idx]
                    print_out(
                        f"Thank you for choice the device id: {device_id}",
                        stdout=True
                    )
                    break
                else:
                    print_out(
                        "Choice out of range, please try again.",
                        stdout=True
                    )
        elif length == 1:
            device_id = devices[0]
            print_out(
                f"current device id: {device_id}",
                stdout=True
            )
        else:
            print_out(
                "No devices found. Exiting.",
                stdout=True,
                log_level="error"
            )
        return device_id
    
    def _get_ability_name(self, 
                          all_package_info: dict) -> str:
        """
        提取主入口 ability 名称。
        """
        hap_modules = all_package_info.get('hapModuleInfos', [])
        main_abilities = []

        for module in hap_modules:
            main_ability = module.get('mainAbility')
            ability_infos = module.get('abilityInfos', [])
            found = None
            for info in ability_infos:
                ability_name = info.get('name')
                if main_ability and main_ability == ability_name:
                    found = ability_name
                    break

            if not main_ability and ability_infos:
                main_abilities.append(ability_infos[0].get('name'))

            if found:
                main_abilities.append(found)

        # 优先返回 EntryAbility
        if 'EntryAbility' in main_abilities:
            return 'EntryAbility'
        # 其次返回包含 mainAbility 或 MainAbility 的
        for name in main_abilities:
            if 'mainAbility' in name or 'MainAbility' in name:
                return name
            
        # 否则返回第一个
        if main_abilities:
            return main_abilities[0]
        else:
            return ''

    def _get_command_list(self,
                          command: str,
                          device_id: Optional[str] = None) -> list:
        command_list = command.split(' ')
        # hdc -> root_dir/hdc/hdc.exe
        command_list[0] = os.path.join(os.environ["ROOT_DIR"], self.hdc_command)
        if device_id:
            command_list.insert(1, '-t')
            command_list.insert(2, device_id)
        return command_list

    def run_hdc_command(self,
                        command: str,
                        device_id: Optional[str] = None) -> CompletedProcess:
        kwargs = {}
        platform_name = platform.system().lower()
        if platform_name == 'windows':
            kwargs.setdefault('creationflags', subprocess.CREATE_NO_WINDOW)
        result = subprocess.run(
            self._get_command_list(command, device_id),
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='ignore',
            **kwargs
        )
        if result.returncode != 0:
            print_out(
                f'run command `{command}` return {result.returncode}',
                log_level="error"
            )
            raise Exception(f'run command `{command}` return {result.returncode}')
        if result.stdout and result.stdout.startswith('[Fail]'):
            print_out(
                f'run command `{command}` error: {result.stdout}',
                log_level="error"
            )
            raise Exception(f'run command `{command}` error: {result.stdout}')
        return result

    def get_connected_devices(self) -> list:
        try:
            result = self.run_hdc_command('hdc list targets')
            devices = []
            for line in result.stdout.splitlines():
                if line.strip() and 'Empty' not in line:
                    devices.append(line.strip())
            return devices
        except Exception as e:
            print_out(
                f'Error getting devices: {e}',
                log_level="error"
            )
            return []

    def get_foreground_app(self) -> str:
        result = self.run_hdc_command('hdc shell aa dump -a', self.device_id)

        pattern = r'app state #FOREGROUND'
        lines = result.stdout.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if 'ExtensionRecords:' in line:
                print_out(
                    f'ExtensionRecords found in aa dump output, skipping parsing',
                    log_level="warning"
                )
                return ''

            if re.search(pattern, line):
                bundle_name = None

                j = i
                while j >= 0 and not (re.search(r'bundle name', lines[j]) or re.search(r'app name', lines[j])):
                    j -= 1

                if j >= 0:
                    for k in range(j, i + 1):
                        l = lines[k].strip()
                        if re.search(r'bundle name', l):
                            bundle_name = re.search(r'\[(.*?)\]', l).group(1)

                if bundle_name:
                    return bundle_name
            i += 1

        return ''

    def get_installed_apps(self) -> list:
        try:
            result = self.run_hdc_command(f'hdc shell bm dump -a', self.device_id)
            apps = []
            for line in result.stdout.splitlines():
                if line.startswith('ID:'):
                    continue
                package_name = line.strip()
                apps.append(package_name)
            return apps
        except Exception as e:
            print_out(
                f"Error getting apps: {e}",
                log_level="error"
            )
            return []

    def get_package_info(self,
                         package_name: str) -> dict:
        app_name = self.bundle_name_dict.get(package_name)
        if not app_name:
            print_out(
                f'Package name {package_name} not found in white list',
                log_level="error"
            )
            return {}

        result = self.run_hdc_command(f'hdc shell bm dump -n "{package_name}"', self.device_id)
        matches = re.findall(f'{package_name}:' + r'([\s\S]*)', result.stdout)
        all_package_info = json.loads(matches[0])

        application_info = all_package_info.get('applicationInfo')
        package_info = {
            'appName': app_name,
            'packageName': package_name,
            'appVersion': application_info.get('versionName'),
            'isSystemApp': application_info.get('isSystemApp'),
            'mainAbility': self._get_ability_name(all_package_info),
        }
        return package_info

    def start_app(self,
                  package_name: str,
                  ability_name: Optional[str] = None, 
                  restart: bool = True) -> None:
        if ability_name is None:
            package_info = self.get_package_info(package_name)
            ability_name = package_info.get('mainAbility')
            if not package_info or not ability_name:
                print_out(
                    f'Failed to start app {package_name}',
                    log_level="error"
                )
                return

        if restart:
            self.run_hdc_command(f'hdc shell aa force-stop "{package_name}"', self.device_id)
            time.sleep(0.1)
        self.run_hdc_command(f'hdc shell aa start -a "{ability_name}" -b "{package_name}"', self.device_id)
        time.sleep(0.1)

    def get_screenshot_data(self) -> tuple[str, str]:
        uid = uuid.uuid4().hex
        screenshot_path = '/data/local/tmp/' + uid + '.jpeg'
        if not os.path.exists(os.environ['TEMP_DIR']):
            os.makedirs(os.environ['TEMP_DIR'])
        local_screenshot_path = os.path.join(os.environ['TEMP_DIR'], uid + '.jpeg')
        self.run_hdc_command(f'hdc shell snapshot_display -f {screenshot_path}', self.device_id)
        self.run_hdc_command(f'hdc file recv {screenshot_path} "{local_screenshot_path}"', self.device_id)
        self.run_hdc_command(f'hdc shell rm {screenshot_path}', self.device_id)

        with Image.open(local_screenshot_path) as img:
            fmt = img.format.upper()
            img = img.resize((int(img.width * self.factor), int(img.height * self.factor)), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            encoded_image = encode_image(byte_stream=buffer.getvalue())
        os.remove(local_screenshot_path)
        return encoded_image, fmt

    def perform_back(self) -> None:
        self.run_hdc_command(f'hdc shell uinput -K -d 2 -u 2', self.device_id)
    
    def perform_home(self) -> None:
        self.run_hdc_command(f'hdc shell uinput -K -d 1 -u 1', self.device_id)

    def perform_click(self,
                      x: Union[int, str], 
                      y: Union[int, str]) -> None:
        self.run_hdc_command(f'hdc shell uinput -T -c {x} {y}', self.device_id)

    def perform_longclick(self,
                          x: Union[int, str], 
                          y: Union[int, str]) -> None:
        self.run_hdc_command(f'hdc shell uinput -T -d {x} {y} -i 1000 -u {x} {y}', self.device_id)

    def perform_scroll(self,
                       x1: Union[int, str], 
                       y1: Union[int, str], 
                       x2: Union[int, str], 
                       y2: Union[int, str]) -> None:
        self.run_hdc_command(f'hdc shell uinput -T -m {x1} {y1} {x2} {y2} 500', self.device_id)

    def perform_settext(self,
                        text: str, 
                        enter: bool = False) -> None:
        self.driver.input_text(text)
        # self.run_hdc_command(f'hdc shell uitest uiInput inputText 1 1 "{text}"', self.device_id)
        if enter:
            self.run_hdc_command(f'hdc shell uinput -K -d 2054 -u 2054', self.device_id)

    def get_screen_scale(self) -> tuple[float, float]:
        result = self.run_hdc_command(f'hdc shell snapshot_display /data/local/tmp/', self.device_id)
        ret = result.stdout

        width_phone = int(re.search(r'width.*\s(\d+)\s*,\s*height.*\s(\d+)\n', ret).group(1))
        height_phone = int(re.search(r'width.*\s(\d+)\s*,\s*height.*\s(\d+)\n', ret).group(2))
        return width_phone, height_phone

    def dump_ui_tree(self,
                     dump_times: int) -> None:
        try:
            result = self.run_hdc_command(f"hdc shell uitest dumpLayout", self.device_id)
            res = result.stdout
            device_full_path = re.search(r'saved to:(.+)\s*\n?', res).group(1)
        except Exception as e:
            print_out(f"UI automator output vide: {e}", log_level="error")
            return

        local_tree_dir = os.path.join(os.environ['DATA_DIR'], "JsonInfo")
        local_tree_dir = os.path.join(local_tree_dir, f'frame_{dump_times}')
        os.makedirs(local_tree_dir, exist_ok=True)
        local_tree_path = os.path.join(local_tree_dir, f'tree_origin.json')
        self.run_hdc_command(f'hdc file recv {device_full_path} "{local_tree_path}"', self.device_id)
        self.run_hdc_command(f"hdc shell rm {device_full_path}", self.device_id)
