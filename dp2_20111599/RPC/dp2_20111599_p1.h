/*
 * Please do not edit this file.
 * It was generated using rpcgen.
 */

#ifndef _DP2_20111599_P1_H_RPCGEN
#define _DP2_20111599_P1_H_RPCGEN

#include <rpc/rpc.h>

#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DP2_20111599_P1_RPC 0x20000000
#define DP2_20111599_P1_RPC_VER 1

#if defined(__STDC__) || defined(__cplusplus)
#define GET_TIME 1
extern  enum clnt_stat get_time_1(void *, int *, CLIENT *);
extern  bool_t get_time_1_svc(void *, int *, struct svc_req *);
#define DELAY 2
extern  enum clnt_stat delay_1(int *, int *, CLIENT *);
extern  bool_t delay_1_svc(int *, int *, struct svc_req *);
extern int dp2_20111599_p1_rpc_1_freeresult (SVCXPRT *, xdrproc_t, caddr_t);

#else /* K&R C */
#define GET_TIME 1
extern  enum clnt_stat get_time_1();
extern  bool_t get_time_1_svc();
#define DELAY 2
extern  enum clnt_stat delay_1();
extern  bool_t delay_1_svc();
extern int dp2_20111599_p1_rpc_1_freeresult ();
#endif /* K&R C */

#ifdef __cplusplus
}
#endif

#endif /* !_DP2_20111599_P1_H_RPCGEN */
