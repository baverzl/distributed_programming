/*
 * This is sample code generated by rpcgen.
 * These are only templates and you can use them
 * as a guideline for developing your own functions.
 */


#include "dp2_20111599_p1.h"

// headers.
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
//extern globals.
int tod;
pthread_rwlock_t rwlock;

bool_t
get_time_1_svc(void *argp, int *result, struct svc_req *rqstp)
{
    bool_t retval;
    pthread_rwlock_rdlock(&rwlock);
    *result = tod;
    pthread_rwlock_unlock(&rwlock);
    retval = TRUE;
    return retval;
}

bool_t
delay_1_svc(int *argp, int *result, struct svc_req *rqstp)
{
    bool_t retval;
    int s_tod, c_tod;
    printf("thread delays %ds\n", *argp);
    pthread_rwlock_rdlock(&rwlock);
    s_tod = tod;
    pthread_rwlock_unlock(&rwlock);
    while (1)
    {
        pthread_rwlock_rdlock(&rwlock);
        c_tod = tod;
        pthread_rwlock_unlock(&rwlock);
        if (s_tod + *argp <= c_tod)
            break;
    }
    *result = c_tod;
    retval = TRUE;
    return retval;
}

int
dp2_20111599_p1_rpc_1_freeresult (SVCXPRT *transp, xdrproc_t xdr_result, caddr_t result)
{
    xdr_free (xdr_result, result);

    /*
     * Insert additional freeing code here, if needed
     */

    return 1;
}
