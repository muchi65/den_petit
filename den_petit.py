import numpy as np
import random
import time
from ctypes import windll
# タイマー精度を1msec単位にする
windll.winmm.timeBeginPeriod(1)

from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

import cv2
from PIL import Image, ImageDraw, ImageFont

import glob

import configparser
ini = configparser.ConfigParser()
ini.read('setting.ini', 'UTF-8')
font = ini["general"]["font"]

# https://qiita.com/mo256man/items/f07bffcf1cfedf0e42e0
def cv2_putText(img, text, org, fontFace, fontScale, color, mode=None, anchor=None):
    """
    mode:
        0:left bottom, 1:left ascender, 2:middle middle,
        3:left top, 4:left baseline
    anchor:
        lb:left bottom, la:left ascender, mm: middle middle,
        lt:left top, ls:left baseline
    """

    # テキスト描画域を取得
    x, y = org
    fontPIL = ImageFont.truetype(font = fontFace, size = fontScale)
    dummy_draw = ImageDraw.Draw(Image.new("L", (0,0)))
    xL, yT, xR, yB = dummy_draw.multiline_textbbox((x, y), text, font=fontPIL)

    # modeおよびanchorによる座標の変換
    img_h, img_w,_ = img.shape
    if mode is None and anchor is None:
        offset_x, offset_y = xL - x, yB - y
    elif mode == 0 or anchor == "lb":
        offset_x, offset_y = xL - x, yB - y
    elif mode == 1 or anchor == "la":
        offset_x, offset_y = 0, 0
    elif mode == 2 or anchor == "mm":
        offset_x, offset_y = (xR + xL)//2 - x, (yB + yT)//2 - y
    elif mode == 3 or anchor == "lt":
        offset_x, offset_y = xL - x, yT - y
    elif mode == 4 or anchor == "ls":
        _, descent = ImageFont.FreeTypeFont(fontFace, fontScale).getmetrics()
        offset_x, offset_y = xL - x, yB - y - descent

    x0, y0 = x - offset_x, y - offset_y
    xL, yT = xL - offset_x, yT - offset_y
    xR, yB = xR - offset_x, yB - offset_y

    # バウンディングボックスを描画　不要ならコメントアウトする
    # cv2.rectangle(img, (xL,yT), (xR,yB), color, 1)
    # 画面外なら何もしない
    if xR<=0 or xL>=img_w or yB<=0 or yT>=img_h:
        print("out of bounds")
        return img

    # ROIを取得する
    x1, y1 = max([xL, 0]), max([yT, 0])
    x2, y2 = min([xR, img_w]), min([yB, img_h])
    roi = img[y1:y2, x1:x2]

    # ROIをPIL化してテキスト描画しCV2に戻る
    roiPIL = Image.fromarray(roi)
    draw = ImageDraw.Draw(roiPIL)
    draw.text((x0-x1, y0-y1), text, color, fontPIL)
    roi = np.array(roiPIL, dtype=np.uint8)
    img[y1:y2, x1:x2] = roi

    return img

def dbg_show(img, name = "dbg", wait=1):
    cv2.imshow(name,cv2.resize(img,None,fx=0.25,fy=0.25))
    cv2.waitKey(wait)

# 日本語ファイル読み込み対応
def imread(filename, flags=-1, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

# アルファチャンネルを分離して読み込み
def imread_split(fname):
    t = imread(fname)
    return t[:,:,:3],t[:,:,3]>0

# フォルダからメモ化読み込み
org_imgs = dict()
org_masks=dict()
def load_image(tgt):
    if tgt in org_imgs:
        return org_imgs[tgt], org_masks[tgt]
    img = []
    mask = []
    for fname in glob.glob(ini[tgt]["image"] +"/*"):
        t = imread(fname)
        img.append(t[:,:,:3])
        mask.append(t[:,:,3]>0)
    org_imgs[tgt] = img
    org_masks[tgt] = mask
    return img,mask

def mask_draw(tgt, img, mask, x, y):
    # アルファチャンネルは0/1としてのマスクとしか使わない
    h,w,_ = img.shape
    x = int(x)
    y = int(y)
    tgt[y:y+h,x:x+w][mask] = img[mask]

class Enemy:
    def __init__(self, name, player):
        self.img, self.mask = load_image(name)
        self.img_num = len(self.img)
        self.h,self.w,_ = self.img[0].shape
        self.y = int(ini['general']['floor_y']) - self.h
        self.x = 0
        self.index=0
        self.speed = int(ini[name]['speed'])
        self.motion_cnt = 0
        self.motion_max = int(ini[name]['motion_max'])
        self.pop_se = pygame.mixer.Sound(ini[name]['pop_se'])
        self.dead_se = pygame.mixer.Sound(ini[name]['dead_se'])
        self.enable=False
        self.level = 1
        self.player = player
        self.score=int(ini[name]['score'])

    def pop(self, x, level):
        self.x = int(x)
        self.level = level
        self.pop_se.play()
        self.enable=True

    def dead(self):
        self.enable = False
        self.dead_se.play()

    def draw(self, img):
        if not self.enable:
            return
        mask_draw(img,self.img[self.index],self.mask[self.index],self.x,self.y)

    def update(self):
        if not self.enable:
            return
        move = self.speed*(1 + self.level/10.0)
        self.x -= move
        self.motion_cnt += move
        if self.motion_cnt >= self.motion_max:
            self.motion_cnt = 0
            self.index+=1
            if self.index == self.img_num:
                self.index = 0
        if self.player.hit(self.x,self.w,-move,self.score):
            self.dead()


class LeeLee:
    def __init__(self):
        sec = "leelee"
        self.img,self.mask = load_image(sec)
        self.x =  int(ini[sec]["x"]) 
        self.y = int(ini[sec]['floor_y']) - self.img[0].shape[0]
        self.motion_cnt = 0
        self.motion_max = int(ini[sec]['motion_max'])
        self.index = 0
        self.img_num = len(self.img)

    def draw(self, img):
        mask_draw(img,self.img[self.index],self.mask[self.index],self.x,self.y)

    def update(self):
        self.motion_cnt += 1
        if self.motion_cnt >= self.motion_max:
            self.motion_cnt = 0
            self.index+=1
            if self.index == self.img_num:
                self.index = 0

class Field:
    def __init__(self):
        self.bg = imread("./image/bg.png")[:,:,:3]
        tgt,mask = imread_split("./image/target.png")
        x = int(ini["leelee"]["target_x"]) 
        y = int(ini["general"]["floor_y"]) - tgt.shape[0] 

        # 背景で動かないものは先に描画を済ませる
        mask_draw(self.bg, tgt,mask,x,y)
        h = int(ini["field"]["height"])
        self.bg = self.bg[:h].copy()
        cv2_putText(self.bg,"◀シャオランの家",(0,y-50),font,12,(255,255,255),anchor="lt")

        self.leelee = LeeLee()

    def update(self):
        # 参照渡しになるのでcopyしないとオリジナルが上書きされる
        bg_temp = self.bg.copy()
        self.leelee.update()
        self.leelee.draw(bg_temp)
        return bg_temp


class Status:
    def __init__(self, h, w):
        self.lv_pos = (10, 10)
        self.fontscale = 20
        self.color =(255,255,255)

    def draw(self, frame, level):
        cv2_putText(frame, "Lv." + str(int(level)),self.lv_pos,font,self.fontscale,self.color,anchor="lt")

class Player:
    def __init__(self):
        self.img,self.mask = imread_split("./image/goo.png")
        self.enable=False
        self.h, self.w, _ = self.img.shape
        self.x = int(ini["leelee"]["target_x"]) 
        self.y = int(ini["general"]["floor_y"]) - self.h
        self.cnt = -1
        self.cooltime =int(ini["leelee"]["cooltime"])
        self.attacktime = self.cooltime // 2
        self.kill = 0
        self.score = 0
        self.attack_se = pygame.mixer.Sound(ini["leelee"]['attack_se'])

    def update(self, attack_cmd=False):
        if self.cnt == -1:
            if attack_cmd:
                self.enable = True
                self.cnt = 0
                self.attack_se.play()
            else:
                return
        else:
            # クールタイム計算
            self.cnt += 1
            if self.cnt == self.attacktime:
                self.enable = False
            elif self.cnt == self.cooltime:
                self.cnt = -1

    def draw(self,img):
        if not self.enable:
            return
        mask_draw(img,self.img,self.mask,self.x,self.y)
    
    def hit(self, moved_x, w, move, score):
        # 敵が高速で移動しても最低1フレームの当たり判定はあるはず
        if not self.enable:
            return False
        min_x = moved_x
        max_x = moved_x - move + w

        is_hit = min_x < (self.x + self.w) and \
           self.x < max_x
        if is_hit:
            self.score += score
            self.kill += 1
        return is_hit

def main():
    pygame.mixer.pre_init(44100, 16, 2, 1024*4)
    pygame.init()
    pygame.mixer.set_num_channels(8) 
    pygame.mixer.music.load(ini["general"]["bgm0"])
    pygame.mixer.music.play(-1)
    finish_se = pygame.mixer.Sound(ini["general"]["finish_se"])
    start_se = pygame.mixer.Sound(ini["general"]["start_se"])
    player = Player()
    enemies = []
    enemies.append(Enemy("denji",player))
    enemies.append(Enemy("osiri",player))
    enemy_num = len(enemies)
    level = 0

    field = Field()
    frame = field.update()
    frame_h, frame_w, _ = frame.shape

    win_name="でんじぷち"
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    logo,logo_mask = imread_split("./image/logo.png")
    key = -1
    while key < 0:
        frame = field.update()
        mask_draw(frame,logo,logo_mask,(frame_w - logo.shape[1])//2, 50)
        cv2.imshow(win_name,frame)
        key = cv2.waitKey(20)

    start_se.play()
    pygame.mixer.music.stop()
    pygame.mixer.music.load(ini["general"]["bgm1"])
    pygame.mixer.music.play(-1)
    pygame.mixer.music.set_volume(0.3)
    frame = field.update()
    img = frame.copy()
    img[frame_h//2-50:frame_h//2+50]=0
    frame = (frame/2.0 + img/2.0).astype(np.uint8)
    cv2_putText(frame, "シャオランを守れ！", (frame_w//2,frame_h//2),font,64,(255,255,255),anchor="mm")
    cv2.imshow(win_name,frame)
    cv2.waitKey(1500)
    status = Status(frame_h, frame_w)

    def pop_enemy(level):
        i = random.randint(0,enemy_num-1)
        enemy = enemies[i]
        enemy.pop(frame_w - enemy.w, level)
        return enemy
    main_timer = time.perf_counter()
    proc_time=0
    enemy = enemies[0]

    while True:
        start = time.perf_counter()

        # 敵が生きてれば移動 いなければpopさせる
        if enemy.enable:
            enemy.update()
        else:
            level+=1
            enemy = pop_enemy(level)
            if level == 30:
                pygame.mixer.music.stop()
                pygame.mixer.music.load(ini["general"]["bgm2"])
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play(-1)

        # 端っこに到達したらゲームオーバー
        if enemy.x < 0:
            finish_se.play()
            h2,w2,_ = frame.shape
            h2 = h2//2
            w2 = w2//2
            txt="Score:" + str(player.score)
            img = frame.copy()
            img[h2-50:h2+50]=0
            frame = (frame/2.0 + img/2.0).astype(np.uint8)
            cv2_putText(frame,txt , (w2 + 3,h2 + 3),font,64,(32,32,32),anchor="mm")
            cv2_putText(frame, txt, (w2,h2),font,64,(255,255,255),anchor="mm")
            cv2.imshow(win_name, frame)
            cv2.waitKey()
            break

        frame = field.update()
        status.draw(frame,level)
        enemy.draw(frame)
        player.draw(frame)

        cv2.imshow(win_name, frame)
        key = cv2.waitKeyEx(1)
        if key == 27:
            break
        player.update(key > 0)

        end = time.perf_counter()
        proc_time = end - main_timer
        time_sec = end - start

        # FPSの制御
        if 0.033 >= time_sec:
            time.sleep(0.033 - time_sec)


if __name__ == "__main__":
    main()