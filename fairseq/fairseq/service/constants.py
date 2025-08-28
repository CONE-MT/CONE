# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

MAX_SEQ_LEN = 256
BATCH_SIZE = 128  # silly high bc we dynamically batch by MAX_BATCH_TOKENS
MAX_BATCH_TOKENS = 256
DEFAULT_PORT = 6020
MODEL_PARALLEL = 1
TOTAL_WORLD_SIZE = 1

try:
    # internal logic denoting where checkpoints are in meta infrastructure
    from metaseq_internal.constants import CHECKPOINT_FOLDER
except ImportError:
    CHECKPOINT_FOLDER = "/app/model_files/"  # specific model path for cone demo



LAUNCH_ARGS = [
    "--task multilingual_translation_branch",
    "--bpe sentencepiece",
    "--lang-pairs aa-trg1,ab-trg1,ace-trg1,acm-trg1,acu-trg1,ady-trg1,ae-trg1,af-trg1,afb-trg1,afh-trg1,agr-trg1,ain-trg1,ak-trg1,ake-trg1,akl-trg1,aln-trg1,am-trg1,amu-trg1,an-trg1,ang-trg1,aoz-trg1,apc-trg1,ar-trg1,arq-trg1,ary-trg1,arz-trg1,as-trg1,ase-trg1,ast-trg1,av-trg1,avk-trg1,awa-trg1,ay-trg1,az-trg1,azb-trg1,ba-trg1,bal-trg1,bar-trg1,be-trg1,bem-trg1,ber-trg1,bg-trg1,bh-trg1,bho-trg1,bi-trg1,bjn-trg1,bm-trg1,bn-trg1,bnt-trg1,bo-trg1,bom-trg1,br-trg1,brx-trg1,bs-trg1,bsn-trg1,bua-trg1,bug-trg1,bvy-trg1,bzt-trg1,ca-trg1,cak-trg1,cb-trg1,cbk-trg1,cdo-trg1,ce-trg1,ceb-trg1,ch-trg1,cho-trg1,chq-trg1,chr-trg1,cjp-trg1,ckb-trg1,cku-trg1,cmn-trg1,cnh-trg1,cni-trg1,co-trg1,cop-trg1,cr-trg1,crh-trg1,crp-trg1,cs-trg1,csb-trg1,cu-trg1,cv-trg1,cx-trg1,cy-trg1,cycl-trg1,da-trg1,de-trg1,dik-trg1,din-trg1,dje-trg1,djk-trg1,dng-trg1,dop-trg1,drt-trg1,dsb-trg1,dtp-trg1,dws-trg1,dz-trg1,ee-trg1,efi-trg1,egl-trg1,el-trg1,en-trg1,enm-trg1,eo-trg1,es-trg1,et-trg1,eu-trg1,evn-trg1,ext-trg1,fa-trg1,ff-trg1,fi-trg1,fil-trg1,fj-trg1,fkv-trg1,fo-trg1,fr-trg1,frm-trg1,fro-trg1,frp-trg1,frr-trg1,fur-trg1,fuv-trg1,fy-trg1,ga-trg1,gan-trg1,gbi-trg1,gbm-trg1,gcf-trg1,gd-trg1,gil-trg1,gl-trg1,gn-trg1,gom-trg1,gos-trg1,got-trg1,gr-trg1,grc-trg1,gsw-trg1,gu-trg1,gv-trg1,ha-trg1,hai-trg1,hak-trg1,haw-trg1,hbo-trg1,hbs-trg1,he-trg1,hi-trg1,hif-trg1,hil-trg1,hne-trg1,ho-trg1,hoc-trg1,hr-trg1,hrx-trg1,hsb-trg1,ht-trg1,hu-trg1,hup-trg1,hus-trg1,hy-trg1,ia-trg1,iba-trg1,id-trg1,ie-trg1,ig-trg1,ik-trg1,ike-trg1,ilo-trg1,inh-trg1,io-trg1,iro-trg1,is-trg1,it-trg1,iu-trg1,izh-trg1,ja-trg1,jak-trg1,jam-trg1,jbo-trg1,jiv-trg1,jv-trg1,ka-trg1,kab-trg1,kam-trg1,kbh-trg1,kek-trg1,kg-trg1,kha-trg1,kik-trg1,kj-trg1,kjh-trg1,kk-trg1,kl-trg1,km-trg1,kmr-trg1,kn-trg1,ko-trg1,koi-trg1,kr-trg1,krl-trg1,ks-trg1,ksh-trg1,ku-trg1,kv-trg1,kw-trg1,ky-trg1,kzj-trg1,la-trg1,lad-trg1,lb-trg1,ldn-trg1,lfn-trg1,lg-trg1,li-trg1,lij-trg1,liv-trg1,lkt-trg1,lld-trg1,lmo-trg1,ln-trg1,lo-trg1,lt-trg1,ltg-trg1,luo-trg1,lut-trg1,luy-trg1,lv-trg1,lzh-trg1,mad-trg1,mai-trg1,mam-trg1,max-trg1,mfe-trg1,mg-trg1,mgm-trg1,mh-trg1,mhr-trg1,mi-trg1,mic-trg1,mik-trg1,min-trg1,miq-trg1,mk-trg1,ml-trg1,mn-trg1,mo-trg1,moh-trg1,mos-trg1,mr-trg1,ms-trg1,mt-trg1,mus-trg1,mvv-trg1,mwl-trg1,mww-trg1,my-trg1,myv-trg1,na-trg1,nah-trg1,nan-trg1,nap-trg1,nb-trg1,nch-trg1,nds-trg1,ne-trg1,ngt-trg1,ngu-trg1,nhg-trg1,niu-trg1,nl-trg1,nlv-trg1,nn-trg1,no-trg1,nog-trg1,non-trg1,nov-trg1,npi-trg1,nr-trg1,ns-trg1,nso-trg1,nst-trg1,nus-trg1,nv-trg1,ny-trg1,oc-trg1,ofs-trg1,ojb-trg1,om-trg1,ood-trg1,or-trg1,orv-trg1,os-trg1,osp-trg1,ota-trg1,pa-trg1,pag-trg1,pam-trg1,pap-trg1,pau-trg1,pcd-trg1,pck-trg1,pdc-trg1,pes-trg1,phn-trg1,pi-trg1,pl-trg1,plt-trg1,pms-trg1,pmy-trg1,pnb-trg1,pot-trg1,ppk-trg1,ppl-trg1,prg-trg1,prs-trg1,ps-trg1,pt-trg1,qa-trg1,qd-trg1,qu-trg1,quc-trg1,que-trg1,quw-trg1,quz-trg1,qya-trg1,rap-trg1,rif-trg1,rm-trg1,rn-trg1,ro-trg1,rom-trg1,ru-trg1,rue-trg1,rup-trg1,rw-trg1,ry-trg1,sa-trg1,sah-trg1,sat-trg1,sc-trg1,scn-trg1,sco-trg1,sd-trg1,sdh-trg1,se-trg1,sg-trg1,sgn-trg1,sgs-trg1,sh-trg1,shi-trg1,shn-trg1,shs-trg1,shy-trg1,si-trg1,sjn-trg1,sk-trg1,sl-trg1,sm-trg1,sma-trg1,sml-trg1,sn-trg1,so-trg1,sq-trg1,sr-trg1,ss-trg1,st-trg1,stq-trg1,su-trg1,sux-trg1,sv-trg1,sw-trg1,swg-trg1,swh-trg1,syr-trg1,sz-trg1,szl-trg1,ta-trg1,tc-trg1,te-trg1,tet-trg1,tg-trg1,th-trg1,thv-trg1,ti-trg1,tk-trg1,tl-trg1,tlh-trg1,tly-trg1,tmh-trg1,tmp-trg1,tmr-trg1,tn-trg1,to-trg1,toki-trg1,tpi-trg1,tpw-trg1,tr-trg1,trv-trg1,ts-trg1,tt-trg1,tvl-trg1,tw-trg1,ty-trg1,tz-trg1,tzl-trg1,udm-trg1,ug-trg1,uk-trg1,umb-trg1,ur-trg1,usp-trg1,uz-trg1,ve-trg1,vec-trg1,vi-trg1,vls-trg1,vo-trg1,wa-trg1,wae-trg1,wal-trg1,war-trg1,wo-trg1,wuu-trg1,xal-trg1,xh-trg1,xmf-trg1,yaq-trg1,yi-trg1,yo-trg1,zam-trg1,ze-trg1,zgh-trg1,zh-trg1,zhs-trg1,zht-trg1,zhtrad-trg1,zhtw-trg1,zhyue-trg1,zlm-trg1,zsm-trg1,zu-trg1,zz-trg1,zza-trg1",
    "--langs  aa,ab,ace,acm,acu,ady,ae,af,afb,afh,agr,ain,ak,ake,akl,aln,am,amu,an,ang,aoz,apc,ar,arq,ary,arz,as,ase,ast,av,avk,awa,ay,az,azb,ba,bal,bar,be,bem,ber,bg,bh,bho,bi,bjn,bm,bn,bnt,bo,bom,br,brx,bs,bsn,bua,bug,bvy,bzt,ca,cak,cb,cbk,cdo,ce,ceb,ch,cho,chq,chr,cjp,ckb,cku,cmn,cnh,cni,co,cop,cr,crh,crp,cs,csb,cu,cv,cx,cy,cycl,da,de,dik,din,dje,djk,dng,dop,drt,dsb,dtp,dws,dz,ee,efi,egl,el,en,enm,eo,es,et,eu,evn,ext,fa,ff,fi,fil,fj,fkv,fo,fr,frm,fro,frp,frr,fur,fuv,fy,ga,gan,gbi,gbm,gcf,gd,gil,gl,gn,gom,gos,got,gr,grc,gsw,gu,gv,ha,hai,hak,haw,hbo,hbs,he,hi,hif,hil,hne,ho,hoc,hr,hrx,hsb,ht,hu,hup,hus,hy,ia,iba,id,ie,ig,ik,ike,ilo,inh,io,iro,is,it,iu,izh,ja,jak,jam,jbo,jiv,jv,ka,kab,kam,kbh,kek,kg,kha,kik,kj,kjh,kk,kl,km,kmr,kn,ko,koi,kr,krl,ks,ksh,ku,kv,kw,ky,kzj,la,lad,lb,ldn,lfn,lg,li,lij,liv,lkt,lld,lmo,ln,lo,lt,ltg,luo,lut,luy,lv,lzh,mad,mai,mam,max,mfe,mg,mgm,mh,mhr,mi,mic,mik,min,miq,mk,ml,mn,mo,moh,mos,mr,ms,mt,mus,mvv,mwl,mww,my,myv,na,nah,nan,nap,nb,nch,nds,ne,ngt,ngu,nhg,niu,nl,nlv,nn,no,nog,non,nov,npi,nr,ns,nso,nst,nus,nv,ny,oc,ofs,ojb,om,ood,or,orv,os,osp,ota,pa,pag,pam,pap,pau,pcd,pck,pdc,pes,phn,pi,pl,plt,pms,pmy,pnb,pot,ppk,ppl,prg,prs,ps,pt,qa,qd,qu,quc,que,quw,quz,qya,rap,rif,rm,rn,ro,rom,ru,rue,rup,rw,ry,sa,sah,sat,sc,scn,sco,sd,sdh,se,sg,sgn,sgs,sh,shi,shn,shs,shy,si,sjn,sk,sl,sm,sma,sml,sn,so,sq,sr,ss,st,stq,su,sux,sv,sw,swg,swh,syr,sz,szl,ta,tc,te,tet,tg,th,thv,ti,tk,tl,tlh,tly,tmh,tmp,tmr,tn,to,toki,tpi,tpw,tr,trv,ts,tt,tvl,tw,ty,tz,tzl,udm,ug,uk,umb,ur,usp,uz,ve,vec,vi,vls,vo,wa,wae,wal,war,wo,wuu,xal,xh,xmf,yaq,yi,yo,zam,ze,zgh,zh,zhs,zht,zhtrad,zhtw,zhyue,zlm,zsm,zu,zz,zza,trg1",
    "--fixed-dictionary  ../fairseq/service/demo_files/merge_dict_nllb ",
    "--gen-subset test",
    "--remove-bpe sentencepiece",
    "--sentencepiece-model  ../fairseq/service/demo_files/flores200sacrebleuspm ",
    "-s en ",
    "-t zh" ,
    f"--path {CHECKPOINT_FOLDER}/checkpoint_last.pt",
    f"--last_ckpt_dir {CHECKPOINT_FOLDER}",
    "--beam 1",
    "--nbest 1",
    "--distributed-port -1",
    "--checkpoint-shard-count 1",
    "--encoder-langtok src ",
    "--decoder-langtok",

    f"--batch-size {BATCH_SIZE}",
    "/tmp",  # required "data" argument.
]
